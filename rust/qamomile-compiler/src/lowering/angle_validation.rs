//! Bounded finite-angle preflight shared by static circuit adapters.
//!
//! Plans reuse the canonical scalar-result representation and native evaluator.
//! Only emitted gate consumers restrict Real values: unused dataflow and
//! non-finite intermediate values retain the ordinary binary64 domain.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use prost::Message;
use thiserror::Error;

use crate::circuit::StaticCircuitValue;
use crate::literals::{
    bytes_literal, bytes_value, int_literal, int_value, string_literal, string_value,
    tuple_literal, tuple_values,
};
use crate::operations::primitive_gate_specs;
use crate::result_postprocessing::decode_plan;
use crate::simulation::CircuitScalarEvaluator;
use crate::{
    CircuitLaunchBinding, CircuitParameter, CircuitScalarId, CircuitScalarNode, StaticCircuit,
    StaticInstruction,
};

const MAX_ANGLE_PLAN_BYTES: usize = 64 * 1024 * 1024;
const MAX_ANGLE_WORK: usize = 1_000_000;
const PLAN_KIND: &str = "qamomile.circuit-angle-validation";

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LoopRange {
    index: u32,
    lower: i64,
    upper: i64,
    stride: i64,
}

impl LoopRange {
    fn count(&self) -> Result<u128, CircuitAngleValidationError> {
        crate::static_values::range_len(self.lower, self.upper, self.stride)
            .ok_or_else(|| invalid("loop range"))
    }
}

/// Rejection before a circuit adapter consumes quantum gate angles.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CircuitAngleValidationError {
    /// A consumed angle is non-finite or its native numeric evaluation fails.
    #[error("Q4482: invalid circuit angle; angles must be finite: {0}")]
    Evaluation(String),
    /// A persistent plan is malformed or exceeds bounded verification work.
    #[error("Q4483: invalid circuit angle validation plan: {0}")]
    InvalidPlan(String),
    /// The angle needs an execution context unavailable before SDK binding.
    #[error("Q4484: circuit angle requires native runtime validation: {0}")]
    RuntimeDependent(String),
}

struct AngleArena<'a> {
    scalars: Cow<'a, [CircuitScalarNode]>,
    roots: Vec<CircuitScalarId>,
    seen: HashSet<CircuitScalarId>,
    loops: Vec<LoopRange>,
}

struct PendingBlock<'a> {
    instructions: &'a [StaticInstruction],
    scalars: &'a [CircuitScalarNode],
    loops: Vec<LoopRange>,
}

/// Derive reusable native checks for every emitted host-evaluable gate angle.
///
/// Shared gate definitions and loop-independent expressions are visited once.
/// Hidden numeric slots are already checked by their native shot template.
/// Scoped loop coordinates remain compact ranges and are evaluated within one
/// shared work budget. Measurement-dependent angles require a runtime route.
///
/// # Errors
/// Rejects static non-finite angles, unsupported runtime dependencies and
/// excessive projection work before an adapter receives the circuit.
pub fn project_circuit_angle_validation(
    circuit: &StaticCircuit,
) -> Result<Vec<Vec<u8>>, CircuitAngleValidationError> {
    let mut remaining = MAX_ANGLE_WORK;
    let arenas = collect_angle_arenas(circuit, &mut remaining)?;
    let mut plans = Vec::with_capacity(arenas.len());
    let mut bytes = 0_usize;
    for arena in arenas {
        charge(&mut remaining, arena.scalars.len() + arena.roots.len())?;
        let dependencies = loop_dependencies(&arena.scalars);
        let (indexed, independent): (Vec<_>, Vec<_>) = arena
            .roots
            .into_iter()
            .partition(|root| dependencies[root.index() as usize]);
        for roots in [independent, indexed] {
            if roots.is_empty() {
                continue;
            }
            let result = StaticCircuitValue::Aggregate(
                roots.into_iter().map(StaticCircuitValue::Scalar).collect(),
            );
            let plan = StaticCircuit::from_result_plan(
                0,
                arena.scalars.clone().into_owned().into(),
                result,
            );
            let Some((scalars, result)) = plan
                .compact_result_plan()
                .map_err(|error| invalid(&error.to_string()))?
            else {
                return Err(invalid("missing gate angle roots"));
            };
            check_scalar_context(&scalars)?;
            let used_loops = scalars
                .iter()
                .filter_map(|node| match node {
                    CircuitScalarNode::LoopIndex(index) => Some(*index),
                    _ => None,
                })
                .collect::<HashSet<_>>();
            let loops = arena
                .loops
                .iter()
                .filter(|range| used_loops.contains(&range.index))
                .cloned()
                .collect::<Vec<_>>();
            if used_loops.len() != loops.len() {
                return Err(invalid("unscoped loop index"));
            }
            let plan = StaticCircuit::from_result_plan(0, scalars, result);
            check_static_angles(&plan)?;
            let payload = encode_angle_plan(&plan, &loops)?;
            bytes = bytes
                .checked_add(payload.len())
                .ok_or_else(|| invalid("byte budget"))?;
            if bytes > MAX_ANGLE_PLAN_BYTES {
                return Err(invalid("byte budget"));
            }
            if !plan
                .scalars()
                .iter()
                .any(|node| matches!(node, CircuitScalarNode::Parameter { .. }))
            {
                evaluate_plan(&plan, &loops, &[], &mut remaining)?;
            }
            plans.push(payload);
        }
    }
    Ok(plans)
}

fn collect_angle_arenas<'a>(
    circuit: &'a StaticCircuit,
    remaining: &mut usize,
) -> Result<Vec<AngleArena<'a>>, CircuitAngleValidationError> {
    let mut pending = vec![PendingBlock {
        instructions: circuit.instructions(),
        scalars: circuit.scalars(),
        loops: Vec::new(),
    }];
    let mut definitions = HashSet::new();
    let mut arena_indices = HashMap::new();
    let mut arenas: Vec<AngleArena<'_>> = Vec::new();
    while let Some(block) = pending.pop() {
        charge(remaining, block.instructions.len())?;
        for instruction in block.instructions {
            let mut child_loops = block.loops.clone();
            if instruction.opcode() == "for" {
                charge(remaining, block.scalars.len())?;
                let (count, range) = loop_range(instruction, block.scalars)?;
                if count == 0 {
                    continue;
                }
                child_loops.extend(range);
            }
            if consumes_angle(instruction.opcode()) {
                let roots = instruction
                    .parameters()
                    .iter()
                    .filter_map(|parameter| match parameter {
                        CircuitParameter::Scalar(root) => Some(*root),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let [root] = roots.as_slice() else {
                    return Err(invalid("gate angle parameter shape"));
                };
                if matches!(
                    block.scalars.get(root.index() as usize),
                    Some(CircuitScalarNode::HiddenBinding(_))
                ) {
                    continue;
                }
                let identity = (block.scalars.as_ptr() as usize, block.loops.clone());
                let index = *arena_indices.entry(identity).or_insert_with(|| {
                    let index = arenas.len();
                    arenas.push(AngleArena {
                        scalars: Cow::Borrowed(block.scalars),
                        roots: Vec::new(),
                        seen: HashSet::new(),
                        loops: block.loops.clone(),
                    });
                    index
                });
                let arena = &mut arenas[index];
                if arena.seen.insert(*root) {
                    arena.roots.push(*root);
                }
                if instruction.opcode() == "pauli_evolution" {
                    add_pauli_angles(arena, instruction, *root, remaining)?;
                }
            }
            for parameter in instruction.parameters() {
                if let CircuitParameter::Definition(definition) = parameter
                    && definitions.insert((definition.identity(), block.loops.clone()))
                    && let Some(body) = definition.retained_body()
                {
                    pending.push(PendingBlock {
                        instructions: body,
                        scalars: definition.scalars(),
                        loops: block.loops.clone(),
                    });
                }
            }
            pending.extend(
                instruction
                    .regions()
                    .iter()
                    .rev()
                    .map(|region| PendingBlock {
                        instructions: region.instructions(),
                        scalars: block.scalars,
                        loops: child_loops.clone(),
                    }),
            );
        }
    }

    Ok(arenas)
}

fn add_pauli_angles(
    arena: &mut AngleArena<'_>,
    instruction: &StaticInstruction,
    time: CircuitScalarId,
    remaining: &mut usize,
) -> Result<(), CircuitAngleValidationError> {
    let Some(CircuitParameter::Hamiltonian(hamiltonian)) = instruction.parameters().first() else {
        return Err(invalid("Pauli evolution Hamiltonian"));
    };
    let scales = crate::pauli_evolution::pauli_evolution_angle_scales(hamiltonian)
        .map_err(|error| invalid(&error.to_string()))?;
    charge(remaining, scales.len().saturating_mul(3))?;
    let scalars = arena.scalars.to_mut();
    for scale in scales {
        let scale_id = CircuitScalarId::from_index(
            u32::try_from(scalars.len()).map_err(|_| invalid("scalar arena size"))?,
        );
        scalars.push(CircuitScalarNode::RealBits(scale.to_bits()));
        let root = CircuitScalarId::from_index(
            u32::try_from(scalars.len()).map_err(|_| invalid("scalar arena size"))?,
        );
        scalars.push(CircuitScalarNode::Apply {
            opcode: "classical.mul".into(),
            result_type: crate::CircuitScalarType::Real,
            attributes: Box::new([]),
            operands: vec![scale_id, time].into(),
        });
        arena.roots.push(root);
    }
    Ok(())
}

/// Validate final gate angles before any SDK parameter binding or submission.
///
/// Each plan is decoded through the existing canonical scalar-plan boundary.
/// Scalar evaluation is shared with Reference execution and preserves lazy
/// selection; this function never allocates a quantum statevector.
///
/// # Errors
/// Rejects malformed or excessive plans, unavailable runtime context, numeric
/// evaluation failures, and non-finite consumed angles.
pub fn validate_circuit_angles(
    plans: &[Vec<u8>],
    bindings: &[CircuitLaunchBinding],
) -> Result<(), CircuitAngleValidationError> {
    if plans.len() > MAX_ANGLE_WORK {
        return Err(invalid("plan count budget"));
    }
    let bytes = plans
        .iter()
        .try_fold(0_usize, |total, plan| total.checked_add(plan.len()));
    if bytes.is_none_or(|bytes| bytes > MAX_ANGLE_PLAN_BYTES) {
        return Err(invalid("byte budget"));
    }
    let mut remaining = MAX_ANGLE_WORK;
    for payload in plans {
        let (circuit, loops) = decode_angle_plan(payload)?;
        evaluate_plan(&circuit, &loops, bindings, &mut remaining)?;
    }
    Ok(())
}

fn consumes_angle(opcode: &str) -> bool {
    matches!(opcode, "global_phase" | "pauli_evolution")
        || primitive_gate_specs()
            .iter()
            .any(|gate| gate.name() == opcode && gate.has_angle())
}

fn angle_roots(
    circuit: &StaticCircuit,
) -> Result<Vec<CircuitScalarId>, CircuitAngleValidationError> {
    let Some(StaticCircuitValue::Aggregate(values)) = circuit.runtime_result_value() else {
        return Err(invalid("angle root structure"));
    };
    values
        .iter()
        .map(|value| match value {
            StaticCircuitValue::Scalar(root) => Ok(*root),
            _ => Err(invalid("angle root is not scalar")),
        })
        .collect()
}

fn check_scalar_context(scalars: &[CircuitScalarNode]) -> Result<(), CircuitAngleValidationError> {
    for node in scalars {
        let unsupported = match node {
            CircuitScalarNode::Measurement(_) => Some("measurement-dependent angle"),
            CircuitScalarNode::CallValue { .. } => Some("runtime-call-dependent angle"),
            CircuitScalarNode::CurrentShot => Some("logical-shot-dependent angle"),
            CircuitScalarNode::HiddenBinding(_) => Some("unresolved hidden numeric angle"),
            _ => None,
        };
        if let Some(reason) = unsupported {
            return Err(CircuitAngleValidationError::RuntimeDependent(reason.into()));
        }
    }
    Ok(())
}

fn check_static_angles(circuit: &StaticCircuit) -> Result<(), CircuitAngleValidationError> {
    let mut depends_on_parameter = Vec::with_capacity(circuit.scalars().len());
    for node in circuit.scalars() {
        let dependent = matches!(
            node,
            CircuitScalarNode::Parameter { .. } | CircuitScalarNode::LoopIndex(_)
        ) || node.operands().iter().any(|operand| {
            depends_on_parameter
                .get(operand.index() as usize)
                .copied()
                .unwrap_or(true)
        });
        depends_on_parameter.push(dependent);
    }
    let mut evaluator = CircuitScalarEvaluator::new(circuit, &[], &[])
        .map_err(|error| invalid(&error.to_string()))?;
    for root in angle_roots(circuit)? {
        if !depends_on_parameter[root.index() as usize] {
            evaluator
                .angle(root)
                .map_err(|error| CircuitAngleValidationError::Evaluation(error.to_string()))?;
        }
    }
    Ok(())
}

fn loop_range(
    instruction: &StaticInstruction,
    scalars: &[CircuitScalarNode],
) -> Result<(u128, Option<LoopRange>), CircuitAngleValidationError> {
    let bounds = instruction
        .parameters()
        .get(..3)
        .ok_or_else(|| invalid("loop bounds"))?;
    let circuit = StaticCircuit::from_result_plan(0, scalars.into(), StaticCircuitValue::Unit);
    let mut evaluator = CircuitScalarEvaluator::new(&circuit, &[], &[])
        .map_err(|error| invalid(&error.to_string()))?;
    let mut values = [0_i64; 3];
    for (target, parameter) in values.iter_mut().zip(bounds) {
        let CircuitParameter::Scalar(root) = parameter else {
            return Err(invalid("loop bound scalar"));
        };
        let literal = evaluator
            .evaluate(*root)
            .map_err(|error| invalid(&error.to_string()))?;
        let Some(crate::wire::literal_proto::Value::IntValue(value)) = literal.value else {
            return Err(invalid("loop bound integer"));
        };
        *target = value;
    }
    let count = crate::static_values::range_len(values[0], values[1], values[2])
        .ok_or_else(|| invalid("loop range"))?;
    let range = if let Some(CircuitParameter::Scalar(root)) = instruction.parameters().get(3) {
        let Some(CircuitScalarNode::LoopIndex(index)) = scalars.get(root.index() as usize) else {
            return Err(invalid("loop coordinate declaration"));
        };
        Some(LoopRange {
            index: *index,
            lower: values[0],
            upper: values[1],
            stride: values[2],
        })
    } else {
        None
    };
    Ok((count, range))
}

fn loop_dependencies(scalars: &[CircuitScalarNode]) -> Vec<bool> {
    let mut dependencies = Vec::with_capacity(scalars.len());
    for node in scalars {
        dependencies.push(
            matches!(node, CircuitScalarNode::LoopIndex(_))
                || node.operands().iter().any(|root| {
                    dependencies
                        .get(root.index() as usize)
                        .copied()
                        .unwrap_or(true)
                }),
        );
    }
    dependencies
}

// The wrapper binds explicit scoped loop slots to a generic scalar-result plan.
// Integer placeholders are restored to LoopIndex before any scalar evaluation.
fn encode_angle_plan(
    circuit: &StaticCircuit,
    loops: &[LoopRange],
) -> Result<Vec<u8>, CircuitAngleValidationError> {
    let mut scalars = circuit.scalars().to_vec();
    let mut slots = Vec::new();
    for (position, scalar) in scalars.iter_mut().enumerate() {
        if let CircuitScalarNode::LoopIndex(index) = scalar {
            slots.push(tuple_literal([
                int_literal(i64::try_from(position).map_err(|_| invalid("loop scalar position"))?),
                int_literal(i64::from(*index)),
            ]));
            *scalar = CircuitScalarNode::Int(0);
        }
    }
    let result = circuit
        .runtime_result_value()
        .cloned()
        .ok_or_else(|| invalid("angle roots"))?;
    // The arena is already compact. Recompacting after inserting equal zero
    // placeholders could change the scope table's scalar positions.
    let payload = crate::wire::CircuitResultPostprocessingProto {
        schema_major: 1,
        schema_minor: 2,
        measurement_count: 0,
        scalars: scalars
            .iter()
            .map(crate::result_postprocessing::encode_scalar)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| invalid(&error.to_string()))?,
        result: Some(
            crate::result_postprocessing::encode_result(&result)
                .map_err(|error| invalid(&error.to_string()))?,
        ),
    }
    .encode_to_vec();
    Ok(tuple_literal([
        string_literal(PLAN_KIND),
        int_literal(1),
        bytes_literal(&payload),
        tuple_literal(slots),
        tuple_literal(loops.iter().map(|range| {
            tuple_literal([
                int_literal(i64::from(range.index)),
                int_literal(range.lower),
                int_literal(range.upper),
                int_literal(range.stride),
            ])
        })),
    ])
    .encode_to_vec())
}

fn decode_angle_plan(
    payload: &[u8],
) -> Result<(StaticCircuit, Vec<LoopRange>), CircuitAngleValidationError> {
    crate::budget::preflight_literal(payload).map_err(|error| invalid(&format!("{error:?}")))?;
    let literal =
        crate::wire::LiteralProto::decode(payload).map_err(|error| invalid(&error.to_string()))?;
    if literal.encode_to_vec() != payload {
        return Err(invalid("noncanonical wrapper"));
    }
    let fields = tuple_values(&literal).ok_or_else(|| invalid("wrapper structure"))?;
    if fields.len() != 5
        || string_value(&fields[0]) != Some(PLAN_KIND)
        || int_value(&fields[1]) != Some(1)
    {
        return Err(invalid("wrapper identity or version"));
    }
    let payload = bytes_value(&fields[2]).ok_or_else(|| invalid("scalar plan"))?;
    let plan = decode_plan(payload)
        .map_err(|error| invalid(&error.to_string()))?
        .circuit;
    if plan.measurement_count() != 0 {
        return Err(invalid("angle preflight cannot consume measurements"));
    }
    let mut loop_ids = HashSet::new();
    let loops = tuple_values(&fields[4])
        .ok_or_else(|| invalid("loop ranges"))?
        .iter()
        .map(|row| {
            let row = integer_row::<4>(row)?;
            let index = u32::try_from(row[0]).map_err(|_| invalid("loop identifier"))?;
            if !loop_ids.insert(index) {
                return Err(invalid("duplicate loop identifier"));
            }
            let range = LoopRange {
                index,
                lower: row[1],
                upper: row[2],
                stride: row[3],
            };
            if range.count()? == 0 {
                return Err(invalid("empty retained loop range"));
            }
            Ok(range)
        })
        .collect::<Result<Vec<_>, CircuitAngleValidationError>>()?;
    let mut scalars = plan.scalars().to_vec();
    let mut positions = HashSet::new();
    let mut used_loops = HashSet::new();
    for slot in tuple_values(&fields[3]).ok_or_else(|| invalid("loop slots"))? {
        let row = integer_row::<2>(slot)?;
        let position = usize::try_from(row[0]).map_err(|_| invalid("loop scalar position"))?;
        let index = u32::try_from(row[1]).map_err(|_| invalid("loop identifier"))?;
        if !positions.insert(position) || !loop_ids.contains(&index) {
            return Err(invalid("duplicate or unscoped loop slot"));
        }
        let scalar = scalars
            .get_mut(position)
            .ok_or_else(|| invalid("loop scalar position"))?;
        if !matches!(scalar, CircuitScalarNode::Int(0)) {
            return Err(invalid("loop slot placeholder"));
        }
        *scalar = CircuitScalarNode::LoopIndex(index);
        used_loops.insert(index);
    }
    if loop_ids != used_loops {
        return Err(invalid("unused loop range"));
    }
    check_scalar_context(&scalars)?;
    let result = plan
        .runtime_result_value()
        .cloned()
        .ok_or_else(|| invalid("angle roots"))?;
    let circuit = StaticCircuit::from_result_plan(0, scalars.into(), result);
    angle_roots(&circuit)?;
    Ok((circuit, loops))
}

fn integer_row<const N: usize>(
    value: &crate::wire::LiteralProto,
) -> Result<[i64; N], CircuitAngleValidationError> {
    let row = tuple_values(value).ok_or_else(|| invalid("integer row"))?;
    if row.len() != N {
        return Err(invalid("integer row width"));
    }
    let mut result = [0_i64; N];
    for (slot, value) in result.iter_mut().zip(row) {
        *slot = int_value(value).ok_or_else(|| invalid("integer row value"))?;
    }
    Ok(result)
}

fn evaluate_plan(
    circuit: &StaticCircuit,
    loops: &[LoopRange],
    bindings: &[CircuitLaunchBinding],
    remaining: &mut usize,
) -> Result<(), CircuitAngleValidationError> {
    let roots = angle_roots(circuit)?;
    let counts = loops
        .iter()
        .map(LoopRange::count)
        .collect::<Result<Vec<_>, _>>()?;
    let count = counts
        .iter()
        .try_fold(1_u128, |total, count| total.checked_mul(*count))
        .ok_or_else(|| invalid("loop validation work budget"))?;
    let work = count
        .checked_mul((circuit.scalars().len() + roots.len()) as u128)
        .and_then(|work| usize::try_from(work).ok())
        .ok_or_else(|| invalid("loop validation work budget"))?;
    charge(remaining, work)?;
    let mut evaluator = CircuitScalarEvaluator::new(circuit, bindings, &[])
        .map_err(|error| invalid(&error.to_string()))?;
    let mut indices = Vec::with_capacity(loops.len());
    for iteration in 0..count {
        let mut ordinal = iteration;
        indices.clear();
        for (range, count) in loops.iter().zip(&counts).rev() {
            let local = ordinal % count;
            ordinal /= count;
            let value = i128::from(range.lower)
                + i128::from(range.stride)
                    * i128::try_from(local).map_err(|_| invalid("loop coordinate"))?;
            indices.push((
                range.index,
                i64::try_from(value).map_err(|_| invalid("loop coordinate"))?,
            ));
        }
        evaluator.set_loop_indices(&indices);
        for root in &roots {
            evaluator
                .angle(*root)
                .map_err(|error| CircuitAngleValidationError::Evaluation(error.to_string()))?;
        }
    }
    Ok(())
}

fn charge(remaining: &mut usize, work: usize) -> Result<(), CircuitAngleValidationError> {
    *remaining = remaining
        .checked_sub(work)
        .ok_or_else(|| invalid("work budget"))?;
    Ok(())
}

fn invalid(reason: &str) -> CircuitAngleValidationError {
    CircuitAngleValidationError::InvalidPlan(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CircuitLaunchValue, CircuitParameterPath, CircuitScalarType, StaticBlock};

    fn id(index: u32) -> CircuitScalarId {
        CircuitScalarId::test_new(index)
    }

    fn parameter(path: Vec<CircuitParameterPath>) -> CircuitScalarNode {
        CircuitScalarNode::Parameter {
            name: "theta".into(),
            path: path.into(),
            scalar_type: CircuitScalarType::Real,
        }
    }

    fn apply(opcode: &str, operands: &[u32]) -> CircuitScalarNode {
        CircuitScalarNode::Apply {
            opcode: opcode.into(),
            result_type: CircuitScalarType::Real,
            attributes: Box::new([]),
            operands: operands.iter().copied().map(id).collect(),
        }
    }

    fn gate(opcode: &str, root: u32) -> StaticInstruction {
        StaticInstruction::test_new(
            opcode,
            vec![0],
            vec![CircuitParameter::Scalar(id(root))],
            vec![],
            None,
            vec![],
        )
    }

    fn circuit(
        scalars: Vec<CircuitScalarNode>,
        instructions: Vec<StaticInstruction>,
    ) -> StaticCircuit {
        StaticCircuit::test_new(1, 0, instructions, scalars, vec![])
    }

    fn binding(value: f64) -> Vec<CircuitLaunchBinding> {
        vec![CircuitLaunchBinding::new(
            "theta".into(),
            CircuitLaunchValue::RealBits(value.to_bits()),
        )]
    }

    #[test]
    fn rejects_nonfinite_angles_for_every_primitive_consumer() {
        let opcodes = primitive_gate_specs()
            .iter()
            .filter(|gate| gate.has_angle())
            .map(|gate| gate.name())
            .chain(["global_phase"]);
        for opcode in opcodes {
            let plans = project_circuit_angle_validation(&circuit(
                vec![parameter(vec![])],
                vec![gate(opcode, 0)],
            ))
            .expect("symbolic angle plan");
            validate_circuit_angles(&plans, &binding(0.25)).expect("finite angle");
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let error =
                    validate_circuit_angles(&plans, &binding(value)).expect_err("non-finite angle");
                assert!(error.to_string().starts_with("Q4482:"));
                assert!(error.to_string().contains("finite"));
            }
        }
    }

    fn pauli_gate(coefficient: f64, constant: f64) -> StaticInstruction {
        let observable = crate::observable::Observable {
            num_qubits: 1,
            constant: num_complex::Complex64::new(constant, 0.0),
            terms: if coefficient == 0.0 {
                std::collections::BTreeMap::default()
            } else {
                [(
                    vec![(0, crate::observable::Pauli::Z)],
                    num_complex::Complex64::new(coefficient, 0.0),
                )]
                .into()
            },
        }
        .encode()
        .encode_to_vec();
        StaticInstruction::test_new(
            "pauli_evolution",
            vec![0],
            vec![
                CircuitParameter::Hamiltonian(observable.into()),
                CircuitParameter::Scalar(id(0)),
            ],
            vec![],
            None,
            vec![],
        )
    }

    #[test]
    fn validates_retained_pauli_effective_rotations_and_global_phase() {
        for (coefficient, constant) in [(2.0, 0.0), (0.0, -2.0)] {
            let plans = project_circuit_angle_validation(&circuit(
                vec![parameter(vec![])],
                vec![pauli_gate(coefficient, constant)],
            ))
            .expect("retained Pauli plan");
            validate_circuit_angles(&plans, &binding(0.25)).expect("finite effective angle");
            validate_circuit_angles(&plans, &binding(0.0)).expect("zero time");
            for time in [f64::MAX, f64::INFINITY, f64::NAN] {
                assert!(matches!(
                    validate_circuit_angles(&plans, &binding(time)),
                    Err(CircuitAngleValidationError::Evaluation(_))
                ));
            }
            assert!(matches!(
                project_circuit_angle_validation(&circuit(
                    vec![CircuitScalarNode::RealBits(f64::MAX.to_bits())],
                    vec![pauli_gate(coefficient, constant)]
                )),
                Err(CircuitAngleValidationError::Evaluation(_))
            ));
        }
        let zero = project_circuit_angle_validation(&circuit(
            vec![parameter(vec![])],
            vec![pauli_gate(0.0, 0.0)],
        ))
        .expect("zero Hamiltonian");
        validate_circuit_angles(&zero, &binding(f64::MAX)).expect("no effective rotation");
        let overflow = project_circuit_angle_validation(&circuit(
            vec![parameter(vec![])],
            vec![pauli_gate(f64::MAX, 0.0)],
        ))
        .expect("coefficient scale is not itself a consumed angle");
        assert!(matches!(
            validate_circuit_angles(&overflow, &binding(0.0)),
            Err(CircuitAngleValidationError::Evaluation(_))
        ));
    }

    #[test]
    fn rejects_static_nonfinite_before_sdk_projection() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                project_circuit_angle_validation(&circuit(
                    vec![CircuitScalarNode::RealBits(value.to_bits())],
                    vec![gate("ry", 0)]
                )),
                Err(CircuitAngleValidationError::Evaluation(_))
            ));
        }
    }

    #[test]
    fn checks_final_expressions_and_preserves_unused_real_data() {
        let finite = circuit(
            vec![
                parameter(vec![]),
                CircuitScalarNode::RealBits(1.0_f64.to_bits()),
                apply("classical.truediv", &[1, 0]),
            ],
            vec![gate("ry", 2)],
        );
        let plans = project_circuit_angle_validation(&finite).expect("division plan");
        validate_circuit_angles(&plans, &binding(f64::INFINITY)).expect("1 / infinity is finite");
        let overflow = circuit(
            vec![
                parameter(vec![]),
                CircuitScalarNode::RealBits(2.0_f64.to_bits()),
                apply("classical.mul", &[0, 1]),
            ],
            vec![gate("ry", 2)],
        );
        let plans = project_circuit_angle_validation(&overflow).expect("multiplication plan");
        assert!(validate_circuit_angles(&plans, &binding(f64::MAX)).is_err());
        let unused = circuit(
            vec![
                parameter(vec![]),
                CircuitScalarNode::RealBits(0.25_f64.to_bits()),
            ],
            vec![gate("ry", 1)],
        );
        let plans = project_circuit_angle_validation(&unused).expect("constant angle plan");
        validate_circuit_angles(&plans, &binding(f64::NAN)).expect("unused non-finite data");
    }

    #[test]
    fn follows_nested_parameter_paths_and_lazy_selection() {
        let scalars = vec![
            parameter(vec![
                CircuitParameterPath::Index(1),
                CircuitParameterPath::Field("angle".into()),
            ]),
            CircuitScalarNode::Bool(false),
            CircuitScalarNode::RealBits(0.0_f64.to_bits()),
            apply("classical.select", &[1, 0, 2]),
        ];
        let values = vec![CircuitLaunchBinding::new(
            "theta".into(),
            CircuitLaunchValue::Aggregate(
                vec![
                    CircuitLaunchValue::RealBits(0.0_f64.to_bits()),
                    CircuitLaunchValue::Record(
                        vec![(
                            "angle".into(),
                            CircuitLaunchValue::RealBits(f64::NAN.to_bits()),
                        )]
                        .into(),
                    ),
                ]
                .into(),
            ),
        )];
        let direct =
            project_circuit_angle_validation(&circuit(scalars.clone(), vec![gate("ry", 0)]))
                .expect("nested plan");
        assert!(validate_circuit_angles(&direct, &values).is_err());
        let lazy = project_circuit_angle_validation(&circuit(scalars, vec![gate("ry", 3)]))
            .expect("lazy plan");
        validate_circuit_angles(&lazy, &values).expect("unselected non-finite branch");
    }

    fn loop_circuit(upper: i64, root: u32) -> StaticCircuit {
        circuit(
            vec![
                CircuitScalarNode::Int(0),
                CircuitScalarNode::Int(upper),
                CircuitScalarNode::Int(1),
                CircuitScalarNode::LoopIndex(7),
                parameter(vec![]),
                apply("classical.mul", &[3, 4]),
            ],
            vec![StaticInstruction::test_new(
                "for",
                vec![],
                vec![
                    CircuitParameter::Scalar(id(0)),
                    CircuitParameter::Scalar(id(1)),
                    CircuitParameter::Scalar(id(2)),
                    CircuitParameter::Scalar(id(3)),
                ],
                vec![],
                None,
                vec![StaticBlock::test_new(vec![gate("ry", root)])],
            )],
        )
    }

    #[test]
    fn validates_retained_loop_angles_without_expanding_metadata() {
        let small = project_circuit_angle_validation(&loop_circuit(3, 5)).expect("small loop plan");
        let larger =
            project_circuit_angle_validation(&loop_circuit(300, 5)).expect("larger loop plan");
        assert!(larger[0].len() <= small[0].len() + 2);
        validate_circuit_angles(&small, &binding(0.25)).expect("finite loop angles");
        assert!(validate_circuit_angles(&small, &binding(f64::MAX)).is_err());
        let empty = project_circuit_angle_validation(&loop_circuit(0, 5)).expect("empty loop");
        assert!(empty.is_empty());
    }

    #[test]
    fn hoists_independent_angles_and_bounds_indexed_validation_work() {
        let independent = project_circuit_angle_validation(&loop_circuit(i64::MAX, 4))
            .expect("independent loop plan");
        validate_circuit_angles(&independent, &binding(0.25)).expect("one finite check");
        let indexed = project_circuit_angle_validation(&loop_circuit(i64::MAX, 5))
            .expect("bounded compact metadata");
        assert!(matches!(
            validate_circuit_angles(&indexed, &binding(0.25)),
            Err(CircuitAngleValidationError::InvalidPlan(_))
        ));
    }

    #[test]
    fn checks_every_nested_loop_coordinate_with_negative_steps() {
        let scalars = vec![
            parameter(vec![]),
            CircuitScalarNode::LoopIndex(3),
            CircuitScalarNode::LoopIndex(9),
            apply("classical.add", &[0, 1]),
            apply("classical.add", &[3, 2]),
            CircuitScalarNode::RealBits(1.0_f64.to_bits()),
            apply("classical.truediv", &[5, 4]),
        ];
        let circuit = StaticCircuit::from_result_plan(
            0,
            scalars.into(),
            StaticCircuitValue::Aggregate(vec![StaticCircuitValue::Scalar(id(6))].into()),
        );
        let plan = encode_angle_plan(
            &circuit,
            &[
                LoopRange {
                    index: 3,
                    lower: 2,
                    upper: -3,
                    stride: -1,
                },
                LoopRange {
                    index: 9,
                    lower: -1,
                    upper: 2,
                    stride: 1,
                },
            ],
        )
        .expect("nested scoped plan");
        validate_circuit_angles(std::slice::from_ref(&plan), &binding(0.5))
            .expect("all coordinates finite");
        assert!(matches!(
            validate_circuit_angles(&[plan], &binding(0.0)),
            Err(CircuitAngleValidationError::Evaluation(_))
        ));
    }

    #[test]
    fn requires_runtime_context_and_delegates_checked_hidden_slots() {
        for scalar in [
            CircuitScalarNode::CurrentShot,
            CircuitScalarNode::Measurement(0),
        ] {
            assert!(matches!(
                project_circuit_angle_validation(&circuit(vec![scalar], vec![gate("ry", 0)])),
                Err(CircuitAngleValidationError::RuntimeDependent(_))
            ));
        }
        let plans = project_circuit_angle_validation(&circuit(
            vec![CircuitScalarNode::HiddenBinding(0)],
            vec![gate("ry", 0)],
        ))
        .expect("native shot binding");
        assert!(plans.is_empty());
    }

    #[test]
    fn rejects_malformed_persistent_angle_plans() {
        for payload in [
            vec![],
            vec![0xff],
            tuple_literal([string_literal("wrong")]).encode_to_vec(),
        ] {
            assert!(matches!(
                validate_circuit_angles(&[payload], &[]),
                Err(CircuitAngleValidationError::InvalidPlan(_))
            ));
        }
        let circuit = loop_circuit(3, 5);
        let mut plans = project_circuit_angle_validation(&circuit).expect("angle plan");
        plans[0].push(0);
        assert!(validate_circuit_angles(&plans, &binding(0.25)).is_err());
    }
}
