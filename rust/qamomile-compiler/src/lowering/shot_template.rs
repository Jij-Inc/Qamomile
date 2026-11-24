//! Fixed-topology current-shot templates and bounded native numeric launch rows.
//!
//! Invocation preflight shares one work bound across numeric launch evaluation
//! and retained-result reconstruction, including encoded ABI output bits and
//! result-only templates. A separate cell bound limits retained numeric rows
//! independently of scalar expression sharing. Checked rows retain flat numeric
//! cells plus cardinality, so zero-slot launches allocate no per-shot storage.

use prost::Message;
use thiserror::Error;

use crate::circuit::{CircuitScalarEvaluatorPlan, lift_shot_circuit};
use crate::simulation::CircuitScalarEvaluator;
use crate::{
    CircuitLaunchBinding, CircuitScalarId, LogicalShotIndex, MAX_INDEXED_SHOTS, StaticCircuit,
    StaticCircuitError,
};

/// Numeric cells retained across all rows of one host-indexed invocation.
const MAX_INDEXED_BINDING_CELLS: u128 = 4_000_000;

/// A compiler-selected current-shot realization.
#[derive(Clone, Debug)]
pub enum ShotCircuitPlan {
    /// One fixed circuit with compiler-owned hidden numeric binding slots.
    Template(Box<ShotCircuitTemplate>),
    /// A topology-dependent source recipe, with its checked projection reason.
    Recipe(String),
}

/// Failure to project, bind, or restore a fixed-topology shot template.
#[derive(Debug, Error)]
pub enum ShotTemplateError {
    /// Native projection failed for a reason other than unsupported topology.
    #[error(transparent)]
    Projection(#[from] StaticCircuitError),
    /// Numeric or persistent template input violated its contract.
    #[error("Q4481: invalid native shot template: {0}")]
    Invalid(String),
}

/// One numeric gate expression in its original independent scalar arena.
#[derive(Clone, Debug)]
pub(crate) struct ShotBindingSlot {
    pub(crate) arena: usize,
    pub(crate) root: CircuitScalarId,
}

/// Immutable fixed circuit plus native scalar evaluation plans.
#[derive(Clone, Debug)]
pub struct ShotCircuitTemplate {
    circuit: StaticCircuit,
    arenas: Vec<CircuitScalarEvaluatorPlan>,
    slots: Vec<ShotBindingSlot>,
    result_plan: Vec<u8>,
    result_work: u128,
}

/// Checked logical cardinality and flat numeric cells in compiler slot order.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotLaunchRows {
    row_count: usize,
    slot_count: usize,
    values: Vec<f64>,
}

impl ShotLaunchRows {
    /// Return checked logical cardinality, including rows with no numeric cells.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Return the uniform compiler-owned width of every row.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Borrow one checked row without allocating, or return None past cardinality.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f64]> {
        if index >= self.row_count {
            return None;
        }
        let start = index * self.slot_count;
        Some(&self.values[start..start + self.slot_count])
    }
}

impl ShotCircuitTemplate {
    /// Borrow the single SDK emission view.
    #[must_use]
    pub const fn circuit(&self) -> &StaticCircuit {
        &self.circuit
    }

    /// Return the number of compiler-owned numeric binding slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Check invocation work and row storage without allocating numeric rows.
    ///
    /// # Errors
    /// Rejects invalid counts, excessive aggregate launch/result work, and
    /// excessive numeric binding storage.
    pub fn check_invocation(&self, shots: u64) -> Result<(), ShotTemplateError> {
        if shots == 0
            || shots > MAX_INDEXED_SHOTS
            || self
                .arenas
                .iter()
                .map(|arena| arena.scalars().len() as u128)
                .sum::<u128>()
                .saturating_add(self.result_work)
                .saturating_mul(u128::from(shots))
                > crate::core::shot_context::MAX_INDEXED_WORK
        {
            return Err(ShotTemplateError::Invalid(
                "launch exceeds its native scalar work budget".into(),
            ));
        }
        if (self.slots.len() as u128).saturating_mul(u128::from(shots)) > MAX_INDEXED_BINDING_CELLS
        {
            return Err(ShotTemplateError::Invalid(
                "launch exceeds its native numeric binding cell budget".into(),
            ));
        }
        Ok(())
    }

    /// Evaluate every row before an adapter can submit any work.
    ///
    /// # Errors
    /// Rejects invalid counts, excessive aggregate launch/result work or numeric
    /// storage, missing bindings, invalid indexing, and non-finite gate angles.
    pub fn bind_rows(
        &self,
        bindings: &[CircuitLaunchBinding],
        shots: u64,
    ) -> Result<ShotLaunchRows, ShotTemplateError> {
        self.check_invocation(shots)?;
        let row_count = usize::try_from(shots).map_err(|_| {
            ShotTemplateError::Invalid("shot count exceeds host index domain".into())
        })?;
        let slot_count = self.slots.len();
        let mut values =
            Vec::with_capacity(row_count.checked_mul(slot_count).ok_or_else(|| {
                ShotTemplateError::Invalid("binding cells exceed host index domain".into())
            })?);
        for index in 0..shots {
            let index = LogicalShotIndex::new(index, shots)
                .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
            let mut evaluators = self
                .arenas
                .iter()
                .map(|arena| CircuitScalarEvaluator::for_index(arena, bindings, index))
                .collect::<Vec<_>>();
            for (arena, evaluator) in self.arenas.iter().zip(evaluators.iter_mut()) {
                if let Some(crate::circuit::StaticCircuitValue::Aggregate(checks)) =
                    arena.runtime_result_value()
                {
                    for check in checks {
                        let crate::circuit::StaticCircuitValue::Scalar(root) = check else {
                            return Err(ShotTemplateError::Invalid(
                                "launch check structure".into(),
                            ));
                        };
                        evaluator
                            .evaluate(*root)
                            .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
                    }
                }
            }
            for slot in &self.slots {
                let value = evaluators[slot.arena]
                    .evaluate(slot.root)
                    .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
                let Some(crate::wire::literal_proto::Value::RealValue(value)) = value.value else {
                    return Err(ShotTemplateError::Invalid(
                        "hidden numeric slot is not Real".into(),
                    ));
                };
                if !value.is_finite() {
                    return Err(ShotTemplateError::Invalid("non-finite gate angle".into()));
                }
                values.push(value);
            }
        }
        Ok(ShotLaunchRows {
            row_count,
            slot_count,
            values,
        })
    }
}

/// Select a native template only when circuit projection certifies its topology.
///
/// # Errors
/// Numeric, selector, overflow, and verification failures are never swallowed as
/// a recipe choice.
pub fn project(
    projected: Result<StaticCircuit, StaticCircuitError>,
    sampled_width: u32,
) -> Result<ShotCircuitPlan, ShotTemplateError> {
    let circuit = match projected {
        Ok(circuit) => circuit,
        Err(
            error @ (StaticCircuitError::Unsupported(_)
            | StaticCircuitError::DynamicClassicalAggregateIndex
            | StaticCircuitError::HostDispatch(_)
            | StaticCircuitError::LaunchMaterialization(_)),
        ) => return Ok(ShotCircuitPlan::Recipe(error.to_string())),
        Err(error) => return Err(error.into()),
    };
    if circuit.has_ordered_template_checks() {
        return Err(ShotTemplateError::Invalid(
            "classical checks require ordered in-job execution".into(),
        ));
    }
    match lift_shot_circuit(circuit) {
        Ok((circuit, arenas, slots)) => {
            let result_plan = crate::circuit::indexed_result_postprocessing(&circuit)
                .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
            let result_work = crate::result_postprocessing::decode_plan(&result_plan)
                .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?
                .per_row_work(sampled_width as usize);
            Ok(ShotCircuitPlan::Template(Box::new(ShotCircuitTemplate {
                circuit,
                arenas,
                slots,
                result_plan,
                result_work,
            })))
        }
        Err(error @ StaticCircuitError::Unsupported(_)) => {
            Ok(ShotCircuitPlan::Recipe(error.to_string()))
        }
        Err(error) => Err(error.into()),
    }
}

impl ShotCircuitTemplate {
    /// Return a portable native evaluator for indexed logical results.
    ///
    /// # Errors
    /// Rejects a runtime-local result that a static adapter cannot evaluate.
    pub fn result_plan(&self) -> Result<Vec<u8>, ShotTemplateError> {
        Ok(self.result_plan.clone())
    }

    /// Persist scalar arenas and slots bound to the enclosing artifact identity.
    ///
    /// # Errors
    /// Rejects unrepresentable scalar nodes or missing logical result plans.
    pub fn to_bytes(
        &self,
        parameter_abi: &[u8],
        result_abi: &[u8],
        target_fingerprint: &str,
        source_digest: &str,
    ) -> Result<Vec<u8>, ShotTemplateError> {
        use crate::wire;
        let arenas = self
            .arenas
            .iter()
            .map(|arena| {
                Ok(wire::CircuitResultPostprocessingProto {
                    schema_major: 1,
                    schema_minor: 2,
                    measurement_count: arena.measurement_count(),
                    scalars: arena
                        .scalars()
                        .iter()
                        .map(crate::result_postprocessing::encode_scalar)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?,
                    result: Some(
                        crate::result_postprocessing::encode_result(
                            arena.runtime_result_value().ok_or_else(|| {
                                ShotTemplateError::Invalid("launch check plan".into())
                            })?,
                        )
                        .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, ShotTemplateError>>()?;
        Ok(wire::ShotCircuitTemplateProto {
            schema_major: 1,
            arenas,
            slots: self
                .slots
                .iter()
                .map(|slot| {
                    Ok(wire::ShotBindingSlotProto {
                        arena: u32::try_from(slot.arena).map_err(|_| {
                            ShotTemplateError::Invalid("arena index exceeds wire domain".into())
                        })?,
                        root: slot.root.index(),
                    })
                })
                .collect::<Result<Vec<_>, ShotTemplateError>>()?,
            result_plan: self.result_plan()?,
            parameter_abi: parameter_abi.to_vec(),
            result_abi: result_abi.to_vec(),
            target_fingerprint: target_fingerprint.into(),
            source_digest: source_digest.into(),
        }
        .encode_to_vec())
    }

    /// Restore numeric evaluation without source compilation or SDK emission.
    ///
    /// # Errors
    /// Rejects noncanonical plans, mismatched identities, forged scalar references,
    /// unsupported scalar types, invalid result ABIs, and host roots containing
    /// trajectory-local data.
    pub fn from_bytes(
        payload: &[u8],
        parameter_abi: &[u8],
        result_abi: &[u8],
        target_fingerprint: &str,
        source_digest: &str,
    ) -> Result<Self, ShotTemplateError> {
        use crate::wire;
        let invalid = |message: &str| ShotTemplateError::Invalid(message.into());
        if payload.len() > 64 * 1024 * 1024 {
            return Err(invalid("persistent plan byte budget"));
        }
        crate::budget::preflight_shot_template(payload).map_err(|error| match error {
            crate::budget::PreflightError::Budget(reason)
            | crate::budget::PreflightError::Malformed(reason) => {
                ShotTemplateError::Invalid(reason)
            }
        })?;
        let plan = wire::ShotCircuitTemplateProto::decode(payload)
            .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
        if plan.encode_to_vec() != payload || plan.schema_major != 1 {
            return Err(invalid("noncanonical or unsupported template schema"));
        }
        if plan.parameter_abi != parameter_abi
            || plan.result_abi != result_abi
            || plan.target_fingerprint != target_fingerprint
            || plan.source_digest != source_digest
        {
            return Err(invalid(
                "template identity disagrees with its enclosing artifact",
            ));
        }
        if plan.arenas.len() > 1_000_000
            || plan.slots.len() > 1_000_000
            || plan
                .arenas
                .iter()
                .map(|arena| arena.scalars.len())
                .sum::<usize>()
                > 1_000_000
        {
            return Err(invalid("persistent scalar work budget"));
        }
        let arenas = decode_arenas(&plan.arenas)?;
        let slots = decode_slots(plan.slots, &arenas)?;
        let result = crate::result_postprocessing::decode_plan(&plan.result_plan)
            .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
        let result_abi = crate::result_codec::checked_result_abi(result_abi)
            .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?;
        let result_work = result.per_row_work(result_abi.logical_width());
        Ok(Self {
            circuit: result.circuit,
            arenas,
            slots,
            result_plan: plan.result_plan,
            result_work,
        })
    }
}

fn decode_arenas(
    encoded: &[crate::wire::CircuitResultPostprocessingProto],
) -> Result<Vec<CircuitScalarEvaluatorPlan>, ShotTemplateError> {
    let invalid = |message: &str| ShotTemplateError::Invalid(message.into());
    encoded
        .iter()
        .map(|arena| {
            if arena.schema_major != 1 || arena.schema_minor != 2 {
                return Err(invalid("unsupported scalar arena schema"));
            }
            let circuit = crate::result_postprocessing::decode_plan(&arena.encode_to_vec())
                .map_err(|error| ShotTemplateError::Invalid(error.to_string()))?
                .circuit;
            match circuit.runtime_result_value() {
                Some(crate::circuit::StaticCircuitValue::Unit) => {}
                Some(crate::circuit::StaticCircuitValue::Aggregate(checks)) => {
                    for check in checks {
                        let crate::circuit::StaticCircuitValue::Scalar(root) = check else {
                            return Err(invalid("invalid launch check root"));
                        };
                        if !host_root(*root, circuit.scalars()) {
                            return Err(invalid("launch check requires trajectory-local state"));
                        }
                    }
                }
                _ => return Err(invalid("invalid launch check arena result")),
            }
            Ok(circuit)
        })
        .collect::<Result<Vec<_>, ShotTemplateError>>()
}

fn decode_slots(
    encoded: Vec<crate::wire::ShotBindingSlotProto>,
    arenas: &[CircuitScalarEvaluatorPlan],
) -> Result<Vec<ShotBindingSlot>, ShotTemplateError> {
    use crate::{CircuitScalarNode, CircuitScalarType};
    use std::collections::HashSet;
    let invalid = |message: &str| ShotTemplateError::Invalid(message.into());
    let mut unique = HashSet::new();
    let mut slots = Vec::with_capacity(encoded.len());
    for slot in encoded {
        let arena = arenas
            .get(slot.arena as usize)
            .ok_or_else(|| invalid("slot arena reference"))?;
        let root = CircuitScalarId::from_index(slot.root);
        if !unique.insert((slot.arena, slot.root))
            || arena
                .scalars()
                .get(slot.root as usize)
                .map(CircuitScalarNode::scalar_type)
                != Some(CircuitScalarType::Real)
        {
            return Err(invalid("duplicate slot or non-real root"));
        }
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        let mut current_shot = false;
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            let node = &arena.scalars()[id.index() as usize];
            match node {
                CircuitScalarNode::CurrentShot => current_shot = true,
                CircuitScalarNode::Measurement(_)
                | CircuitScalarNode::CallValue { .. }
                | CircuitScalarNode::LoopIndex(_)
                | CircuitScalarNode::HiddenBinding(_) => {
                    return Err(invalid("non-host template root"));
                }
                _ => pending.extend(node.operands()),
            }
        }
        if !current_shot {
            return Err(invalid("hidden slot lacks current-shot dependency"));
        }
        slots.push(ShotBindingSlot {
            arena: slot.arena as usize,
            root,
        });
    }
    Ok(slots)
}

fn host_root(root: CircuitScalarId, scalars: &[crate::CircuitScalarNode]) -> bool {
    use std::collections::HashSet;
    let mut pending = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = scalars.get(id.index() as usize) else {
            return false;
        };
        match node {
            crate::CircuitScalarNode::Measurement(_)
            | crate::CircuitScalarNode::CallValue { .. }
            | crate::CircuitScalarNode::LoopIndex(_)
            | crate::CircuitScalarNode::HiddenBinding(_) => return false,
            _ => pending.extend(node.operands()),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::StaticCircuitValue;
    use crate::{CircuitParameter, CircuitScalarNode, CircuitScalarType, StaticInstruction, wire};

    fn bool_result_abi() -> Vec<u8> {
        wire::ResultAbiProto {
            per_shot: false,
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Scalar.into(),
                name: "Bool".into(),
                ..Default::default()
            }),
            source_order: vec![0],
            native_width: 1,
        }
        .encode_to_vec()
    }

    #[test]
    fn encoded_result_width_is_checked_before_launch_and_after_restore() {
        let width = 65_536;
        let result_abi = wire::ResultAbiProto {
            per_shot: false,
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "UInt".into(),
                arguments: vec![wire::TypeProto {
                    kind: wire::type_proto::Kind::Scalar.into(),
                    name: "Bit".into(),
                    ..Default::default()
                }],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(u64::from(width))),
                }],
                ..Default::default()
            }),
            source_order: (0..width).collect(),
            native_width: width,
        }
        .encode_to_vec();
        let circuit = StaticCircuit::test_new_with_result(
            0,
            0,
            vec![],
            vec![CircuitScalarNode::Int(0)],
            StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)),
        );
        let ShotCircuitPlan::Template(template) = project(Ok(circuit), width).unwrap() else {
            panic!("constant result has fixed topology");
        };
        let restored = ShotCircuitTemplate::from_bytes(
            &template
                .to_bytes(&[], &result_abi, "target", "source")
                .unwrap(),
            &[],
            &result_abi,
            "target",
            "source",
        )
        .expect("restore checked output width");
        for template in [&*template, &restored] {
            assert!(template.bind_rows(&[], 1).is_ok());
            assert!(matches!(
                template.bind_rows(&[], 4096),
                Err(ShotTemplateError::Invalid(reason)) if reason.contains("work budget")
            ));
        }
    }

    fn indexed_angles() -> ShotCircuitTemplate {
        let scalars = vec![
            CircuitScalarNode::CurrentShot,
            CircuitScalarNode::RealBits(0.0f64.to_bits()),
            CircuitScalarNode::RealBits(std::f64::consts::PI.to_bits()),
            CircuitScalarNode::Indexed {
                index: CircuitScalarId::test_new(0),
                elements: vec![CircuitScalarId::test_new(1), CircuitScalarId::test_new(2)]
                    .into_boxed_slice(),
                scalar_type: CircuitScalarType::Real,
            },
            CircuitScalarNode::Bool(true),
        ];
        let circuit = StaticCircuit::test_new_with_result(
            1,
            0,
            vec![StaticInstruction::test_new(
                "ry",
                vec![0],
                vec![CircuitParameter::Scalar(CircuitScalarId::test_new(3))],
                vec![],
                None,
                vec![],
            )],
            scalars,
            StaticCircuitValue::Scalar(CircuitScalarId::test_new(4)),
        );
        let ShotCircuitPlan::Template(template) =
            project(Ok(circuit), 1).expect("native fixed topology")
        else {
            panic!("numeric selection must be a template");
        };
        *template
    }

    fn result_only_template(scalars: Vec<CircuitScalarNode>) -> ShotCircuitTemplate {
        let root = CircuitScalarId::test_new(u32::try_from(scalars.len() - 1).unwrap());
        let circuit = StaticCircuit::test_new_with_result(
            0,
            0,
            vec![],
            scalars,
            StaticCircuitValue::Scalar(root),
        );
        let ShotCircuitPlan::Template(template) = project(Ok(circuit), 1).expect("result template")
        else {
            panic!("result-only context has fixed topology");
        };
        *template
    }

    #[test]
    fn ten_million_shot_preflight_is_allocation_free_and_survives_restore() {
        let template = result_only_template(vec![
            CircuitScalarNode::CurrentShot,
            CircuitScalarNode::Bool(true),
        ]);
        let restored = ShotCircuitTemplate::from_bytes(
            &template
                .to_bytes(&[], &bool_result_abi(), "target", "source")
                .unwrap(),
            &[],
            &bool_result_abi(),
            "target",
            "source",
        )
        .unwrap();
        for template in [&template, &restored] {
            let rows = template
                .bind_rows(&[], 10_000_000)
                .expect("compact checked rows");
            assert_eq!(rows.row_count(), 10_000_000);
            assert_eq!(rows.slot_count(), 0);
            assert_eq!(rows.values.capacity(), 0);
            assert_eq!(rows.row(9_999_999), Some([].as_slice()));
            assert_eq!(rows.row(10_000_000), None);
            assert!(template.check_invocation(10_000_001).is_err());
            assert!(template.bind_rows(&[], 10_000_001).is_err());
            assert!(template.check_invocation(0).is_err());
        }
        assert_eq!(MAX_INDEXED_SHOTS, 10_000_000);
    }

    #[test]
    fn numeric_binding_storage_is_checked_before_rows_are_allocated() {
        for (slots, max_shots) in [(4, 1_000_000), (5, 800_000)] {
            let mut scalars = vec![CircuitScalarNode::CurrentShot];
            let mut instructions = Vec::new();
            for _ in 0..slots {
                let constant = CircuitScalarId::test_new(u32::try_from(scalars.len()).unwrap());
                scalars.push(CircuitScalarNode::RealBits(0.0f64.to_bits()));
                let root = CircuitScalarId::test_new(u32::try_from(scalars.len()).unwrap());
                scalars.push(CircuitScalarNode::Indexed {
                    index: CircuitScalarId::test_new(0),
                    elements: vec![constant].into_boxed_slice(),
                    scalar_type: CircuitScalarType::Real,
                });
                instructions.push(StaticInstruction::test_new(
                    "ry",
                    vec![0],
                    vec![CircuitParameter::Scalar(root)],
                    vec![],
                    None,
                    vec![],
                ));
            }
            let result = CircuitScalarId::test_new(u32::try_from(scalars.len()).unwrap());
            scalars.push(CircuitScalarNode::Bool(true));
            let circuit = StaticCircuit::test_new_with_result(
                1,
                0,
                instructions,
                scalars,
                StaticCircuitValue::Scalar(result),
            );
            let ShotCircuitPlan::Template(template) = project(Ok(circuit), 1).unwrap() else {
                panic!("numeric selection must have fixed topology");
            };
            assert_eq!(template.slot_count(), slots);
            let restored = ShotCircuitTemplate::from_bytes(
                &template
                    .to_bytes(&[], &bool_result_abi(), "target", "source")
                    .unwrap(),
                &[],
                &bool_result_abi(),
                "target",
                "source",
            )
            .unwrap();
            for template in [&*template, &restored] {
                assert!(template.check_invocation(max_shots).is_ok());
                assert!(matches!(
                    template.check_invocation(max_shots + 1),
                    Err(ShotTemplateError::Invalid(reason)) if reason.contains("binding cell budget")
                ));
                assert!(matches!(
                    template.bind_rows(&[], max_shots + 1),
                    Err(ShotTemplateError::Invalid(reason)) if reason.contains("binding cell budget")
                ));
            }
        }
    }

    #[test]
    fn result_only_launch_work_is_checked_before_rows_escape() {
        let mut scalars = vec![CircuitScalarNode::CurrentShot];
        for index in 0..4096 {
            scalars.push(CircuitScalarNode::Apply {
                opcode: "classical.not".into(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![CircuitScalarId::test_new(index)].into_boxed_slice(),
            });
        }
        let template = result_only_template(scalars);
        assert!(
            template
                .arenas
                .iter()
                .all(|arena| arena.scalars().is_empty())
        );
        assert!(matches!(
            template.check_invocation(4096),
            Err(ShotTemplateError::Invalid(reason)) if reason.contains("work budget")
        ));
        assert!(template.bind_rows(&[], 1).is_ok());
        let restored = ShotCircuitTemplate::from_bytes(
            &template
                .to_bytes(&[], &bool_result_abi(), "target", "source")
                .unwrap(),
            &[],
            &bool_result_abi(),
            "target",
            "source",
        )
        .expect("restore result-only work");
        assert!(matches!(
            restored.check_invocation(4096),
            Err(ShotTemplateError::Invalid(reason)) if reason.contains("work budget")
        ));
    }

    #[test]
    fn numeric_and_result_work_share_one_invocation_budget() {
        let template = result_only_template(vec![
            CircuitScalarNode::CurrentShot,
            CircuitScalarNode::Apply {
                opcode: "classical.not".into(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![CircuitScalarId::test_new(0)].into_boxed_slice(),
            },
        ]);
        assert!(template.check_invocation(1_000_000).is_ok());
        let mut persisted = wire::ShotCircuitTemplateProto::decode(
            template
                .to_bytes(&[], &bool_result_abi(), "target", "source")
                .unwrap()
                .as_slice(),
        )
        .unwrap();
        persisted.arenas[0].scalars = vec![
            wire::CircuitResultScalarProto {
                node: Some(wire::circuit_result_scalar_proto::Node::BoolValue(true)),
            };
            31_250
        ];
        let restored = ShotCircuitTemplate::from_bytes(
            &persisted.encode_to_vec(),
            &[],
            &bool_result_abi(),
            "target",
            "source",
        )
        .expect("restore individually bounded launch and result plans");
        assert!(restored.check_invocation(4000).is_ok());
        assert!(matches!(
            restored.check_invocation(4096),
            Err(ShotTemplateError::Invalid(reason)) if reason.contains("work budget")
        ));
    }

    #[test]
    fn result_reconstruction_bounds_aggregates_and_literal_contents() {
        use crate::static_values::{StaticScalar, StaticValue};
        use std::sync::Arc;

        let results = [
            (
                StaticCircuitValue::Aggregate(
                    vec![StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)); 32_768]
                        .into_boxed_slice(),
                ),
                32_768,
            ),
            (
                StaticCircuitValue::Portable(StaticValue::Aggregate(
                    vec![Some(StaticValue::Scalar(StaticScalar::Bool(true))); 32_768].into(),
                )),
                32_768,
            ),
            (
                StaticCircuitValue::Portable(StaticValue::Bytes(Arc::from(vec![0; 32_768]))),
                0,
            ),
            (
                StaticCircuitValue::Portable(StaticValue::String(Arc::from("x".repeat(32_768)))),
                0,
            ),
        ];
        for (result, sampled_width) in results {
            let circuit = StaticCircuit::test_new_with_result(
                0,
                0,
                vec![],
                vec![CircuitScalarNode::Bool(true)],
                result,
            );
            let ShotCircuitPlan::Template(template) = project(Ok(circuit), sampled_width).unwrap()
            else {
                panic!("portable results have fixed topology");
            };
            assert!(template.bind_rows(&[], 1).is_ok());
            assert!(matches!(
                template.check_invocation(4096),
                Err(ShotTemplateError::Invalid(reason)) if reason.contains("work budget")
            ));
        }
    }

    #[test]
    fn unused_loop_nodes_are_excluded_from_persisted_numeric_arenas() {
        let circuit = StaticCircuit::test_new_with_result(
            1,
            0,
            vec![StaticInstruction::test_new(
                "ry",
                vec![0],
                vec![CircuitParameter::Scalar(CircuitScalarId::test_new(4))],
                vec![],
                None,
                vec![],
            )],
            vec![
                CircuitScalarNode::LoopIndex(0),
                CircuitScalarNode::CurrentShot,
                CircuitScalarNode::RealBits(0.0f64.to_bits()),
                CircuitScalarNode::RealBits(std::f64::consts::PI.to_bits()),
                CircuitScalarNode::Indexed {
                    index: CircuitScalarId::test_new(1),
                    elements: vec![CircuitScalarId::test_new(2), CircuitScalarId::test_new(3)]
                        .into_boxed_slice(),
                    scalar_type: CircuitScalarType::Real,
                },
                CircuitScalarNode::Bool(true),
            ],
            StaticCircuitValue::Scalar(CircuitScalarId::test_new(5)),
        );
        let ShotCircuitPlan::Template(template) = project(Ok(circuit), 1).expect("fixed topology")
        else {
            panic!("unrelated loop nodes cannot require a recipe");
        };
        assert_eq!(template.arenas[0].scalars().len(), 4);
        assert_eq!(template.slots[0].root.index(), 3);
        assert!(matches!(
            template.circuit().scalars()[0],
            CircuitScalarNode::LoopIndex(0)
        ));
        let payload = template
            .to_bytes(&[], &bool_result_abi(), "target", "source")
            .expect("persist compact arena");
        let restored =
            ShotCircuitTemplate::from_bytes(&payload, &[], &bool_result_abi(), "target", "source")
                .expect("restore remapped roots");
        assert_eq!(
            restored.bind_rows(&[], 2).expect("rows").values,
            vec![0.0, std::f64::consts::PI]
        );
    }

    #[test]
    fn reachable_loop_index_requires_a_recipe() {
        let circuit = StaticCircuit::test_new_with_result(
            1,
            0,
            vec![StaticInstruction::test_new(
                "ry",
                vec![0],
                vec![CircuitParameter::Scalar(CircuitScalarId::test_new(4))],
                vec![],
                None,
                vec![],
            )],
            vec![
                CircuitScalarNode::LoopIndex(0),
                CircuitScalarNode::CurrentShot,
                CircuitScalarNode::Apply {
                    opcode: "classical.add".into(),
                    result_type: CircuitScalarType::Int,
                    attributes: Box::new([]),
                    operands: vec![CircuitScalarId::test_new(0), CircuitScalarId::test_new(1)]
                        .into_boxed_slice(),
                },
                CircuitScalarNode::RealBits(0.0f64.to_bits()),
                CircuitScalarNode::Indexed {
                    index: CircuitScalarId::test_new(2),
                    elements: vec![CircuitScalarId::test_new(3)].into_boxed_slice(),
                    scalar_type: CircuitScalarType::Real,
                },
                CircuitScalarNode::Bool(true),
            ],
            StaticCircuitValue::Scalar(CircuitScalarId::test_new(5)),
        );
        assert!(matches!(
            project(Ok(circuit), 1),
            Ok(ShotCircuitPlan::Recipe(_))
        ));
    }

    #[test]
    fn binds_fixed_table_once_and_rejects_out_of_bounds_before_rows_escape() {
        let template = indexed_angles();
        assert_eq!(template.slot_count(), 1);
        assert_eq!(
            template.bind_rows(&[], 2).expect("two checked rows").values,
            vec![0.0, std::f64::consts::PI]
        );
        assert!(template.bind_rows(&[], 3).is_err());
        assert!(template.bind_rows(&[], MAX_INDEXED_SHOTS + 1).is_err());
        let CircuitParameter::Scalar(root) = template.circuit().instructions()[0].parameters()[0]
        else {
            panic!("numeric slot");
        };
        assert!(matches!(
            template.circuit().scalars()[root.index() as usize],
            CircuitScalarNode::HiddenBinding(0)
        ));
    }

    #[test]
    fn persisted_slots_reject_forged_references_schema_and_identity() {
        let template = indexed_angles();
        let payload = template
            .to_bytes(&[], &bool_result_abi(), "target", "source")
            .expect("encode native plan");
        let restored =
            ShotCircuitTemplate::from_bytes(&payload, &[], &bool_result_abi(), "target", "source")
                .expect("restore checked numeric plans");
        assert_eq!(
            restored.bind_rows(&[], 2).expect("restored rows"),
            template.bind_rows(&[], 2).expect("original rows")
        );
        assert_eq!(
            restored.result_plan().expect("result plan"),
            template.result_plan().expect("result plan")
        );
        assert!(
            ShotCircuitTemplate::from_bytes(&payload, &[], &bool_result_abi(), "other", "source")
                .is_err()
        );
        for corrupt in [0, 1, 2] {
            let mut forged = wire::ShotCircuitTemplateProto::decode(payload.as_slice())
                .expect("decode test plan");
            match corrupt {
                0 => forged.slots[0].root = u32::MAX,
                1 => forged.slots[0].arena = u32::MAX,
                _ => forged.arenas[0].schema_major = 999,
            }
            assert!(
                ShotCircuitTemplate::from_bytes(
                    &forged.encode_to_vec(),
                    &[],
                    &bool_result_abi(),
                    "target",
                    "source"
                )
                .is_err()
            );
        }
    }
    #[test]
    fn persistent_template_preflight_bounds_empty_messages_and_indexed_operands() {
        let empty_arenas = [0x12, 0x00].repeat(crate::budget::MAX_MESSAGE_NODES + 1);
        assert!(matches!(
            crate::budget::preflight_shot_template(&empty_arenas),
            Err(crate::budget::PreflightError::Budget(_))
        ));
        let indexed = wire::CircuitResultIndexedProto {
            index: 0,
            elements: vec![0; crate::budget::MAX_MESSAGE_NODES + 1],
            r#type: wire::CircuitResultScalarTypeProto::CircuitResultReal.into(),
        };
        let arena = wire::CircuitResultPostprocessingProto {
            schema_major: 1,
            schema_minor: 2,
            measurement_count: 0,
            scalars: vec![wire::CircuitResultScalarProto {
                node: Some(wire::circuit_result_scalar_proto::Node::Indexed(indexed)),
            }],
            result: None,
        };
        let template = wire::ShotCircuitTemplateProto {
            arenas: vec![arena],
            ..Default::default()
        }
        .encode_to_vec();
        assert!(matches!(
            crate::budget::preflight_shot_template(&template),
            Err(crate::budget::PreflightError::Budget(_))
        ));
    }
}
