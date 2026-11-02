//! Shared checked definitions. SDK gate expansion is an explicit consumer action.

use super::{
    Arc, BTreeSet, CircuitScalarNode, CircuitValue, Flow, HashMap, OnceLock, StaticCircuitError,
    StaticInstruction, StaticLowerer, distribute_controls,
};
use crate::lowering::arithmetic_permutation::CheckedArithmeticPermutation;
use crate::lowering::modular_realization::CheckedModularDefinition;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LocalPartition {
    wires: Box<[u64]>,
    widths: Box<[usize]>,
}

type WirePartitions = Vec<LocalPartition>;

#[derive(Debug)]
enum DefinitionSource {
    Body(Box<[StaticInstruction]>),
    Arithmetic(CheckedArithmeticPermutation),
    Modular(CheckedModularDefinition),
}

#[derive(Debug)]
struct DefinitionOwner {
    width: u64,
    scalars: Box<[CircuitScalarNode]>,
    source: DefinitionSource,
    materialized: OnceLock<Result<Box<[StaticInstruction]>, StaticCircuitError>>,
}

/// Immutable process-local owner of a checked, locally addressed definition.
///
/// Identity is scoped to the retained owner, never persisted as compiler meaning.
#[derive(Clone, Debug)]
pub struct CircuitDefinition(Arc<DefinitionOwner>);

impl PartialEq for CircuitDefinition {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for CircuitDefinition {}

impl CircuitDefinition {
    fn new(width: u64, scalars: Box<[CircuitScalarNode]>, source: DefinitionSource) -> Self {
        Self(Arc::new(DefinitionOwner {
            width,
            scalars,
            source,
            materialized: OnceLock::new(),
        }))
    }

    pub(super) fn body(
        width: u64,
        instructions: Vec<StaticInstruction>,
        scalars: Box<[CircuitScalarNode]>,
    ) -> Self {
        Self::new(
            width,
            scalars,
            DefinitionSource::Body(instructions.into_boxed_slice()),
        )
    }

    pub(super) fn arithmetic(width: u64, recipe: CheckedArithmeticPermutation) -> Self {
        Self::new(width, Box::new([]), DefinitionSource::Arithmetic(recipe))
    }

    pub(super) fn modular(width: u64, plan: CheckedModularDefinition) -> Self {
        Self::new(width, Box::new([]), DefinitionSource::Modular(plan))
    }

    /// Borrow an already retained body without materializing an arithmetic recipe.
    pub(super) fn retained_body(&self) -> Option<&[StaticInstruction]> {
        match &self.0.source {
            DefinitionSource::Body(body) => Some(body),
            DefinitionSource::Arithmetic(_) | DefinitionSource::Modular(_) => None,
        }
    }

    /// Return an identity valid only while this owner remains alive.
    #[must_use]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Return the local operand width, including planned clean scratch.
    #[must_use]
    pub fn width(&self) -> u64 {
        self.0.width
    }

    /// Borrow the definition's independent, compact scalar arena.
    #[must_use]
    pub fn scalars(&self) -> &[CircuitScalarNode] {
        &self.0.scalars
    }

    /// Materialize the definition once through the compiler's checked planners.
    ///
    /// # Errors
    /// Returns the original bounded-realization diagnostic on failure.
    pub fn instructions(&self) -> Result<&[StaticInstruction], StaticCircuitError> {
        if let DefinitionSource::Body(body) = &self.0.source {
            return Ok(body);
        }
        self.0
            .materialized
            .get_or_init(|| {
                let instructions = match &self.0.source {
                    DefinitionSource::Arithmetic(recipe) => recipe
                        .materialize()?
                        .into_steps()
                        .into_iter()
                        .map(|step| {
                            debug_assert!(step.angle().is_none());
                            StaticInstruction {
                                opcode: step.gate().as_str().to_owned(),
                                qubits: step.indices().into(),
                                controls: step.controls().into(),
                                parameters: Box::new([]),
                                condition: None,
                                regions: Box::new([]),
                            }
                        })
                        .collect::<Vec<_>>(),
                    DefinitionSource::Modular(plan) => plan
                        .materialize()?
                        .steps()
                        .iter()
                        .map(|step| StaticInstruction {
                            opcode: step.gate().as_str().to_owned(),
                            qubits: step.wires().into(),
                            controls: step.controls().into(),
                            parameters: Box::new([]),
                            condition: None,
                            regions: Box::new([]),
                        })
                        .collect::<Vec<_>>(),
                    DefinitionSource::Body(_) => unreachable!("body returned above"),
                };
                Ok(instructions.into_boxed_slice())
            })
            .as_deref()
            .map_err(Clone::clone)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct CallCacheKey {
    callee: String,
    arguments: Vec<CircuitValue>,
    partitions: WirePartitions,
}

#[derive(Clone)]
pub(super) struct CallCacheEntry {
    instruction: StaticInstruction,
    flow: Flow,
    partitions: WirePartitions,
    scratch_width: usize,
}

pub(super) fn remap_instruction_wires(
    instructions: &mut [StaticInstruction],
    map: &HashMap<u64, u64>,
) -> Result<(), StaticCircuitError> {
    for instruction in instructions {
        remap_wires(&mut instruction.qubits, map)?;
        remap_wires(&mut instruction.controls, map)?;
        for block in &mut instruction.regions {
            remap_instruction_wires(&mut block.instructions, map)?;
        }
    }
    Ok(())
}

fn remap_wires(wires: &mut [u64], map: &HashMap<u64, u64>) -> Result<(), StaticCircuitError> {
    for wire in wires {
        *wire = *map
            .get(wire)
            .ok_or_else(|| StaticCircuitError::Missing("shared definition wire".to_owned()))?;
    }
    Ok(())
}

fn remap_value(
    value: &mut CircuitValue,
    map: &HashMap<u64, u64>,
) -> Result<(), StaticCircuitError> {
    match value {
        CircuitValue::Wires(wires) => remap_wires(wires, map)?,
        CircuitValue::Aggregate(values) => {
            for value in values {
                remap_value(value, map)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn collect_value_wires(value: &CircuitValue, wires: &mut BTreeSet<u64>) {
    match value {
        CircuitValue::Wires(row) => wires.extend(row.iter().copied()),
        CircuitValue::Aggregate(values) => {
            for value in values {
                collect_value_wires(value, wires);
            }
        }
        _ => {}
    }
}

impl StaticLowerer<'_> {
    fn local_partitions(&self, map: &HashMap<u64, u64>) -> WirePartitions {
        let mut partitions = self
            .coherent_partitions
            .iter()
            .filter_map(|(wires, widths)| {
                let wires = wires
                    .iter()
                    .map(|wire| map.get(wire).copied())
                    .collect::<Option<Vec<_>>>()?;
                Some(LocalPartition {
                    wires: wires.into_boxed_slice(),
                    widths: widths.clone(),
                })
            })
            .collect::<Vec<_>>();
        partitions.sort();
        partitions
    }

    pub(super) fn call_cache_key(
        &self,
        callee: &str,
        arguments: &[CircuitValue],
    ) -> Option<(CallCacheKey, Vec<u64>)> {
        let mut wires = BTreeSet::new();
        for value in arguments {
            collect_value_wires(value, &mut wires);
        }
        let wires = wires.into_iter().collect::<Vec<_>>();
        let map = wires
            .iter()
            .copied()
            .enumerate()
            .map(|(index, wire)| (wire, index as u64))
            .collect();
        let mut arguments = arguments.to_vec();
        for value in &mut arguments {
            remap_value(value, &map).ok()?;
        }
        Some((
            CallCacheKey {
                callee: callee.to_owned(),
                arguments,
                partitions: self.local_partitions(&map),
            },
            wires,
        ))
    }

    pub(super) fn cache_call(
        &mut self,
        key: CallCacheKey,
        wires: &[u64],
        start: usize,
        flow: &Flow,
    ) {
        let [instruction] = &self.instructions[start..] else {
            return;
        };
        if !instruction.opcode.starts_with("box:") {
            return;
        }
        let map = wires
            .iter()
            .chain(&self.schoolbook_workspace)
            .copied()
            .enumerate()
            .map(|(index, wire)| (wire, index as u64))
            .collect();
        let mut instruction = instruction.clone();
        instruction.controls = Box::new([]);
        if remap_instruction_wires(std::slice::from_mut(&mut instruction), &map).is_err() {
            return;
        }
        let mut flow = flow.clone();
        for value in &mut flow.values {
            if remap_value(value, &map).is_err() {
                return;
            }
        }
        self.call_cache.insert(
            key,
            CallCacheEntry {
                instruction,
                flow,
                partitions: self.local_partitions(&map),
                scratch_width: self.schoolbook_workspace.len(),
            },
        );
    }

    pub(super) fn replay_cached_call(
        &mut self,
        entry: &CallCacheEntry,
        wires: &[u64],
    ) -> Result<Flow, StaticCircuitError> {
        let scratch = self.schoolbook_workspace(entry.scratch_width)?;
        let map = wires
            .iter()
            .chain(&scratch)
            .copied()
            .enumerate()
            .map(|(index, wire)| (index as u64, wire))
            .collect();
        let mut instruction = entry.instruction.clone();
        remap_instruction_wires(std::slice::from_mut(&mut instruction), &map)?;
        let controls = self.controls.clone();
        distribute_controls(std::slice::from_mut(&mut instruction), &controls)?;
        self.push_instruction(
            &instruction.opcode,
            instruction.qubits,
            instruction.parameters,
        )?;
        for partition in &entry.partitions {
            let mut row = partition.wires.clone();
            remap_wires(&mut row, &map)?;
            self.coherent_partitions
                .insert(row, partition.widths.clone());
        }
        let mut flow = entry.flow.clone();
        for value in &mut flow.values {
            remap_value(value, &map)?;
        }
        Ok(flow)
    }
}

#[cfg(test)]
mod tests {
    use super::CircuitDefinition;
    use crate::lowering::arithmetic_permutation::{
        ArithmeticOperation, CheckedArithmeticPermutation,
    };

    #[test]
    fn arithmetic_owner_is_lazy_shared_and_materializes_once() {
        let recipe =
            CheckedArithmeticPermutation::new(ArithmeticOperation::ModMul, 3, &[2, 5]).unwrap();
        let owner = CircuitDefinition::arithmetic(3, recipe);
        let shared = owner.clone();
        assert!(owner.0.materialized.get().is_none());
        let instructions = owner.instructions().unwrap();
        assert!(!instructions.is_empty());
        assert!(std::ptr::eq(instructions, shared.instructions().unwrap()));
        assert_eq!(owner.identity(), shared.identity());
    }
    #[test]
    fn deferred_arithmetic_rejects_invalid_domains_and_work_before_materialization() {
        assert!(
            CheckedArithmeticPermutation::new(ArithmeticOperation::ModMul, 3, &[0, 5]).is_err()
        );
        assert!(
            CheckedArithmeticPermutation::new(ArithmeticOperation::AddConstant, 1025, &[1])
                .is_err()
        );
        assert!(
            CheckedArithmeticPermutation::new(ArithmeticOperation::MulAdd, 1_000_000, &[]).is_err()
        );
    }
}
