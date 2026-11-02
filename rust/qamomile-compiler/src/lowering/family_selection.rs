//! Canonical launch-time selectors for portable static artifact families.

use std::collections::{BTreeSet, HashSet};

use num_bigint::BigInt;
use prost::Message;
use thiserror::Error;

use crate::budget;
use crate::result_postprocessing::{decode_scalar, encode_scalar};
use crate::simulation::evaluate_dispatch_predicate;
use crate::wire;
use crate::{
    CircuitDispatchDecision, CircuitDispatchParameter, CircuitDispatchPredicate,
    CircuitLaunchBinding, CircuitLaunchValue, CircuitScalarId, CircuitScalarNode,
    CircuitScalarType, StaticCircuitFamily,
};

const PLAN_SCHEMA_MAJOR: u32 = 1;
const PLAN_SCHEMA_MINOR: u32 = 0;
const MAX_PLAN_BYTES: usize = 64 * 1024 * 1024;
const MAX_VARIANTS: usize = 256;
const MAX_PREDICATES: usize = 255;

/// One checked decision path in a persistent artifact-family selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitFamilySelectionVariant {
    decisions: Box<[CircuitDispatchDecision]>,
}

impl CircuitFamilySelectionVariant {
    /// Return the predicate assignments selecting this zero-based variant.
    #[must_use]
    pub fn decisions(&self) -> &[CircuitDispatchDecision] {
        &self.decisions
    }
}

/// A canonical, target-independent launch selector without backend payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitFamilySelectionPlan {
    dispatch_parameters: Box<[CircuitDispatchParameter]>,
    predicates: Box<[CircuitDispatchPredicate]>,
    variants: Box<[CircuitFamilySelectionVariant]>,
}

impl CircuitFamilySelectionPlan {
    /// Return source-ordered entry parameters referenced by the predicates.
    #[must_use]
    pub fn dispatch_parameters(&self) -> &[CircuitDispatchParameter] {
        &self.dispatch_parameters
    }

    /// Return the verified Boolean scalar DAGs used by the decision tree.
    #[must_use]
    pub fn predicates(&self) -> &[CircuitDispatchPredicate] {
        &self.predicates
    }

    /// Return canonical false-first variant paths.
    #[must_use]
    pub fn variants(&self) -> &[CircuitFamilySelectionVariant] {
        &self.variants
    }
}

/// Failure to persist, validate, or evaluate a static-family selector.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CircuitFamilySelectionError {
    /// The persistent selector is malformed, non-canonical, or unsupported.
    #[error("Q4461: invalid native circuit-family selector: {0}")]
    InvalidPlan(String),
    /// Exact launch values cannot evaluate the selected predicate path.
    #[error("Q4462: native circuit-family selection failed: {0}")]
    Evaluation(String),
}

/// Encode the compiler-owned decision tree independently of backend children.
///
/// # Errors
///
/// Returns [`CircuitFamilySelectionError`] if the selector cannot be encoded
/// canonically or violates the persistent plan contract.
pub fn encode_circuit_family_selection(
    family: &StaticCircuitFamily,
) -> Result<Vec<u8>, CircuitFamilySelectionError> {
    let plan = wire::CircuitFamilySelectionProto {
        schema_major: PLAN_SCHEMA_MAJOR,
        schema_minor: PLAN_SCHEMA_MINOR,
        dispatch_parameters: family
            .dispatch_parameters()
            .iter()
            .map(|parameter| {
                Ok(wire::CircuitFamilyDispatchParameterProto {
                    parameter_index: u64::try_from(parameter.parameter_index()).map_err(|_| {
                        invalid("dispatch parameter index exceeds the portable range")
                    })?,
                    name: parameter.name().to_owned(),
                    uint_width: parameter.uint_width().unwrap_or(0),
                })
            })
            .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?,
        predicates: family
            .predicates()
            .iter()
            .map(|predicate| {
                Ok(wire::CircuitFamilyPredicateProto {
                    scalars: predicate
                        .scalars()
                        .iter()
                        .map(encode_scalar)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| invalid(&error.to_string()))?,
                    root: predicate.root().index(),
                })
            })
            .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?,
        variants: family
            .variants()
            .iter()
            .map(|variant| {
                Ok(wire::CircuitFamilyVariantProto {
                    decisions: variant
                        .decisions()
                        .iter()
                        .map(|decision| {
                            Ok(wire::CircuitFamilyDecisionProto {
                                predicate_index: u32::try_from(decision.predicate_index())
                                    .map_err(|_| {
                                        invalid("predicate index exceeds the portable range")
                                    })?,
                                value: decision.value(),
                            })
                        })
                        .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?,
                })
            })
            .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?,
    };
    let payload = plan.encode_to_vec();
    inspect_circuit_family_selection(&payload)?;
    Ok(payload)
}

/// Decode and fully validate one untrusted persistent selector.
///
/// # Errors
///
/// Returns [`CircuitFamilySelectionError`] if the bytes are non-canonical,
/// exceed a structural budget, or do not describe one complete decision tree.
pub fn inspect_circuit_family_selection(
    payload: &[u8],
) -> Result<CircuitFamilySelectionPlan, CircuitFamilySelectionError> {
    if payload.len() > MAX_PLAN_BYTES {
        return Err(invalid("plan exceeds its byte budget"));
    }
    budget::preflight_circuit_family_selection(payload)
        .map_err(|error| invalid(&preflight_reason(error)))?;
    let encoded = wire::CircuitFamilySelectionProto::decode(payload)
        .map_err(|error| invalid(&error.to_string()))?;
    if encoded.encode_to_vec() != payload {
        return Err(invalid("plan is not canonical"));
    }
    if encoded.schema_major != PLAN_SCHEMA_MAJOR || encoded.schema_minor > PLAN_SCHEMA_MINOR {
        return Err(invalid(&format!(
            "unsupported schema {}.{}",
            encoded.schema_major, encoded.schema_minor
        )));
    }
    if encoded.dispatch_parameters.is_empty()
        || encoded.predicates.is_empty()
        || encoded.predicates.len() > MAX_PREDICATES
        || encoded.variants.is_empty()
        || encoded.variants.len() > MAX_VARIANTS
    {
        return Err(invalid("invalid selector dimensions"));
    }

    let dispatch_parameters = decode_dispatch_parameters(&encoded.dispatch_parameters)?;
    let predicates = decode_predicates(&encoded.predicates, &dispatch_parameters)?;
    let variants = decode_variants(&encoded.variants, predicates.len())?;
    validate_decision_tree(&variants)?;
    Ok(CircuitFamilySelectionPlan {
        dispatch_parameters: dispatch_parameters.into_boxed_slice(),
        predicates: predicates.into_boxed_slice(),
        variants: variants.into_boxed_slice(),
    })
}

fn decode_dispatch_parameters(
    encoded: &[wire::CircuitFamilyDispatchParameterProto],
) -> Result<Vec<CircuitDispatchParameter>, CircuitFamilySelectionError> {
    let mut indices = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut previous_index = None;
    encoded
        .iter()
        .map(|parameter| {
            let index = usize::try_from(parameter.parameter_index)
                .map_err(|_| invalid("dispatch parameter index exceeds usize"))?;
            if parameter.name.is_empty()
                || !indices.insert(index)
                || !names.insert(parameter.name.clone())
                || previous_index.is_some_and(|previous| previous >= index)
            {
                return Err(invalid("invalid dispatch parameter order"));
            }
            previous_index = Some(index);
            Ok(CircuitDispatchParameter::from_parts(
                index,
                parameter.name.clone(),
                (parameter.uint_width != 0).then_some(parameter.uint_width),
            ))
        })
        .collect()
}

fn decode_predicates(
    encoded: &[wire::CircuitFamilyPredicateProto],
    parameters: &[CircuitDispatchParameter],
) -> Result<Vec<CircuitDispatchPredicate>, CircuitFamilySelectionError> {
    let predicates = encoded
        .iter()
        .map(|predicate| {
            let scalars = predicate
                .scalars
                .iter()
                .enumerate()
                .map(|(index, scalar)| decode_scalar(scalar, index, 0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| invalid(&error.to_string()))?;
            usize::try_from(predicate.root)
                .ok()
                .and_then(|root| scalars.get(root))
                .filter(|root| root.scalar_type() == CircuitScalarType::Bool)
                .ok_or_else(|| invalid("predicate root is not a valid Bool scalar"))?;
            Ok(CircuitDispatchPredicate::from_parts(
                CircuitScalarId::from_index(predicate.root),
                scalars.into_boxed_slice(),
            ))
        })
        .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?;
    let referenced_names = predicates
        .iter()
        .flat_map(CircuitDispatchPredicate::scalars)
        .filter_map(|scalar| match scalar {
            CircuitScalarNode::Parameter { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let parameter_names = parameters
        .iter()
        .map(|parameter| parameter.name().to_owned())
        .collect::<BTreeSet<_>>();
    if referenced_names != parameter_names {
        return Err(invalid("predicate parameter set disagrees with its ABI"));
    }
    Ok(predicates)
}

fn decode_variants(
    encoded: &[wire::CircuitFamilyVariantProto],
    predicate_count: usize,
) -> Result<Vec<CircuitFamilySelectionVariant>, CircuitFamilySelectionError> {
    encoded
        .iter()
        .map(|variant| {
            let mut seen = HashSet::new();
            let decisions = variant
                .decisions
                .iter()
                .map(|decision| {
                    let index = usize::try_from(decision.predicate_index)
                        .map_err(|_| invalid("predicate index exceeds usize"))?;
                    if index >= predicate_count || !seen.insert(index) {
                        return Err(invalid("invalid or duplicate variant decision"));
                    }
                    Ok(CircuitDispatchDecision::from_parts(index, decision.value))
                })
                .collect::<Result<Vec<_>, CircuitFamilySelectionError>>()?;
            Ok(CircuitFamilySelectionVariant {
                decisions: decisions.into_boxed_slice(),
            })
        })
        .collect()
}

/// Select exactly one variant using the simulator's authoritative scalar semantics.
///
/// # Errors
///
/// Returns [`CircuitFamilySelectionError`] if the selector or launch values are
/// invalid, or if the checked tree does not select exactly one variant.
pub fn select_circuit_family_variant(
    payload: &[u8],
    bindings: &[CircuitLaunchBinding],
) -> Result<usize, CircuitFamilySelectionError> {
    let plan = inspect_circuit_family_selection(payload)?;
    let bindings = normalize_uint_bindings(&plan, bindings)?;
    let mut values = vec![None; plan.predicates.len()];
    let mut selected = None;
    for (variant_index, variant) in plan.variants.iter().enumerate() {
        let mut matches = true;
        for decision in variant.decisions() {
            let predicate_index = decision.predicate_index();
            let actual = if let Some(value) = values[predicate_index] {
                value
            } else {
                let value =
                    evaluate_dispatch_predicate(&plan.predicates[predicate_index], &bindings)
                        .map_err(|error| {
                            CircuitFamilySelectionError::Evaluation(error.to_string())
                        })?;
                values[predicate_index] = Some(value);
                value
            };
            if actual != decision.value() {
                matches = false;
                break;
            }
        }
        if matches && selected.replace(variant_index).is_some() {
            return Err(CircuitFamilySelectionError::Evaluation(
                "selector paths overlap".to_owned(),
            ));
        }
    }
    selected.ok_or_else(|| {
        CircuitFamilySelectionError::Evaluation("no selector path matches".to_owned())
    })
}

fn normalize_uint_bindings(
    plan: &CircuitFamilySelectionPlan,
    bindings: &[CircuitLaunchBinding],
) -> Result<Vec<CircuitLaunchBinding>, CircuitFamilySelectionError> {
    bindings
        .iter()
        .map(|binding| {
            let Some(width) = plan
                .dispatch_parameters()
                .iter()
                .find(|parameter| parameter.name() == binding.name())
                .and_then(CircuitDispatchParameter::uint_width)
            else {
                return Ok(binding.clone());
            };
            let integer = match binding.value() {
                CircuitLaunchValue::Int(value) => BigInt::from(*value),
                CircuitLaunchValue::BigInt(value) => BigInt::from_signed_bytes_be(value),
                _ => {
                    return Err(CircuitFamilySelectionError::Evaluation(format!(
                        "unsigned launch parameter {:?} is not an integer",
                        binding.name()
                    )));
                }
            };
            if integer < BigInt::from(0_u8) || integer.bits() > width {
                return Err(CircuitFamilySelectionError::Evaluation(format!(
                    "unsigned launch parameter {:?} is outside UInt[{width}]",
                    binding.name()
                )));
            }
            let values = (0..width)
                .map(|index| {
                    let shift = width - index - 1;
                    CircuitLaunchValue::Bool(
                        ((&integer >> shift) & BigInt::from(1_u8)) == BigInt::from(1_u8),
                    )
                })
                .collect::<Vec<_>>();
            Ok(CircuitLaunchBinding::new(
                binding.name().to_owned(),
                CircuitLaunchValue::Aggregate(values.into_boxed_slice()),
            ))
        })
        .collect()
}

fn validate_decision_tree(
    variants: &[CircuitFamilySelectionVariant],
) -> Result<(), CircuitFamilySelectionError> {
    let mut pending = vec![(0_usize, variants.len(), 0_usize)];
    while let Some((start, end, depth)) = pending.pop() {
        let paths = &variants[start..end];
        let has_leaf = paths
            .iter()
            .any(|variant| variant.decisions().len() == depth);
        if has_leaf {
            if paths.len() != 1 || paths[0].decisions().len() != depth {
                return Err(invalid("decision tree contains an overlapping leaf"));
            }
            continue;
        }
        let predicate_index = paths[0].decisions()[depth].predicate_index();
        if paths
            .iter()
            .any(|variant| variant.decisions()[depth].predicate_index() != predicate_index)
        {
            return Err(invalid("decision tree changes predicate within a subtree"));
        }
        let split = paths
            .iter()
            .position(|variant| variant.decisions()[depth].value())
            .unwrap_or(paths.len());
        if split == 0
            || split == paths.len()
            || paths[..split]
                .iter()
                .any(|variant| variant.decisions()[depth].value())
            || paths[split..]
                .iter()
                .any(|variant| !variant.decisions()[depth].value())
        {
            return Err(invalid("decision tree is incomplete or non-canonical"));
        }
        pending.push((start + split, end, depth + 1));
        pending.push((start, start + split, depth + 1));
    }
    Ok(())
}

fn invalid(reason: &str) -> CircuitFamilySelectionError {
    CircuitFamilySelectionError::InvalidPlan(reason.to_owned())
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
    use crate::CircuitLaunchValue;

    fn plan_bytes(scalars: &[CircuitScalarNode], root: u32, paths: &[&[(u32, bool)]]) -> Vec<u8> {
        wire::CircuitFamilySelectionProto {
            schema_major: PLAN_SCHEMA_MAJOR,
            schema_minor: PLAN_SCHEMA_MINOR,
            dispatch_parameters: vec![wire::CircuitFamilyDispatchParameterProto {
                parameter_index: 0,
                name: "enabled".to_owned(),
                uint_width: 0,
            }],
            predicates: vec![wire::CircuitFamilyPredicateProto {
                scalars: scalars
                    .iter()
                    .map(encode_scalar)
                    .collect::<Result<_, _>>()
                    .unwrap(),
                root,
            }],
            variants: paths
                .iter()
                .map(|path| wire::CircuitFamilyVariantProto {
                    decisions: path
                        .iter()
                        .map(
                            |(predicate_index, value)| wire::CircuitFamilyDecisionProto {
                                predicate_index: *predicate_index,
                                value: *value,
                            },
                        )
                        .collect(),
                })
                .collect(),
        }
        .encode_to_vec()
    }

    fn enabled_parameter() -> CircuitScalarNode {
        CircuitScalarNode::Parameter {
            name: "enabled".to_owned(),
            path: Box::new([]),
            scalar_type: CircuitScalarType::Bool,
        }
    }

    #[test]
    fn selects_unsigned_word_predicates_from_integer_launch_values() {
        let parameter = CircuitScalarNode::Parameter {
            name: "enabled".to_owned(),
            path: vec![crate::CircuitParameterPath::Index(0)].into_boxed_slice(),
            scalar_type: CircuitScalarType::Bool,
        };
        let mut encoded = wire::CircuitFamilySelectionProto::decode(
            plan_bytes(&[parameter], 0, &[&[(0, false)], &[(0, true)]]).as_slice(),
        )
        .expect("decode test plan");
        encoded.dispatch_parameters[0].uint_width = 3;
        let payload = encoded.encode_to_vec();

        assert_eq!(
            select_circuit_family_variant(
                &payload,
                &[CircuitLaunchBinding::new(
                    "enabled".to_owned(),
                    CircuitLaunchValue::Int(4),
                )],
            ),
            Ok(1),
        );
    }

    #[test]
    fn selector_evaluates_a_deep_scalar_dag_iteratively() {
        let mut scalars = vec![enabled_parameter()];
        for index in 0..20_000_u32 {
            scalars.push(CircuitScalarNode::Apply {
                opcode: "classical.not".to_owned(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![CircuitScalarId::from_index(index)].into_boxed_slice(),
            });
        }
        let payload = plan_bytes(&scalars, 20_000, &[&[(0, false)], &[(0, true)]]);
        let bindings = [CircuitLaunchBinding::new(
            "enabled".to_owned(),
            CircuitLaunchValue::Bool(false),
        )];

        assert_eq!(select_circuit_family_variant(&payload, &bindings), Ok(0));
    }

    #[test]
    fn selector_preserves_lazy_select_semantics() {
        let scalars = vec![
            enabled_parameter(),
            CircuitScalarNode::Bool(true),
            CircuitScalarNode::Int(1),
            CircuitScalarNode::Int(0),
            CircuitScalarNode::Apply {
                opcode: "classical.floordiv".to_owned(),
                result_type: CircuitScalarType::Int,
                attributes: Box::new([]),
                operands: vec![
                    CircuitScalarId::from_index(2),
                    CircuitScalarId::from_index(3),
                ]
                .into_boxed_slice(),
            },
            CircuitScalarNode::Apply {
                opcode: "classical.eq".to_owned(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![
                    CircuitScalarId::from_index(4),
                    CircuitScalarId::from_index(3),
                ]
                .into_boxed_slice(),
            },
            CircuitScalarNode::Apply {
                opcode: "classical.select".to_owned(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![
                    CircuitScalarId::from_index(0),
                    CircuitScalarId::from_index(1),
                    CircuitScalarId::from_index(5),
                ]
                .into_boxed_slice(),
            },
        ];
        let payload = plan_bytes(&scalars, 6, &[&[(0, false)], &[(0, true)]]);
        let bindings = [CircuitLaunchBinding::new(
            "enabled".to_owned(),
            CircuitLaunchValue::Bool(true),
        )];

        assert_eq!(select_circuit_family_variant(&payload, &bindings), Ok(1));
    }

    #[test]
    fn selector_rejects_incomplete_and_noncanonical_plans() {
        let incomplete = plan_bytes(&[enabled_parameter()], 0, &[&[(0, false)]]);
        assert!(matches!(
            inspect_circuit_family_selection(&incomplete),
            Err(CircuitFamilySelectionError::InvalidPlan(_))
        ));

        let mut noncanonical =
            plan_bytes(&[enabled_parameter()], 0, &[&[(0, false)], &[(0, true)]]);
        noncanonical.extend_from_slice(&[0x30, 0x01]);
        assert!(matches!(
            inspect_circuit_family_selection(&noncanonical),
            Err(CircuitFamilySelectionError::InvalidPlan(_))
        ));
    }
}
