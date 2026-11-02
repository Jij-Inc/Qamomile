//! Conservative SSA reuse for endpoints with verified top-level Boolean stores.

use super::{
    CircuitParameter, CircuitScalarId, CircuitScalarNode, CircuitScalarType, HashMap, HashSet,
    RealizationCapabilities, StaticCircuitError, StaticInstruction, StaticLowerer,
};

impl StaticLowerer<'_> {
    pub(super) fn retain_boolean_stores(&mut self) -> Result<(), StaticCircuitError> {
        if !self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::BOOLEAN_STORE)
        {
            return Ok(());
        }
        let mut uses = HashMap::<CircuitScalarId, usize>::new();
        for instruction in &self.instructions {
            if let Some(root) = instruction.condition {
                *uses.entry(root).or_default() += 1;
            }
        }
        let mut available = HashSet::new();
        let mut stored = HashMap::new();
        let mut output = Vec::with_capacity(self.instructions.len());
        for mut instruction in std::mem::take(&mut self.instructions) {
            if let Some(root) = instruction.condition {
                if let Some(replacement) = stored.get(&root) {
                    instruction.condition = Some(*replacement);
                } else if uses.get(&root).copied().unwrap_or(0) > 1
                    && matches!(self.scalars.get(root.index() as usize), Some(CircuitScalarNode::Apply { opcode, result_type: CircuitScalarType::Bool, .. }) if opcode == "classical.select")
                    && boolean_inputs_available(root, &self.scalars, &available)
                {
                    let slot = self.next_measurement;
                    self.next_measurement =
                        slot.checked_add(1).ok_or(StaticCircuitError::Overflow)?;
                    let replacement = self.intern_scalar(CircuitScalarNode::Measurement(slot))?;
                    // Source SSA measurement slots are never overwritten. Only top-level
                    // instructions enter this pass, so the store dominates every reuse.
                    output.push(StaticInstruction {
                        opcode: "store_bool".to_owned(),
                        qubits: Box::new([]),
                        parameters: vec![
                            CircuitParameter::MeasurementIndex(slot),
                            CircuitParameter::Scalar(root),
                        ]
                        .into_boxed_slice(),
                        controls: Box::new([]),
                        condition: None,
                        regions: Box::new([]),
                    });
                    stored.insert(root, replacement);
                    instruction.condition = Some(replacement);
                }
            }
            if matches!(
                instruction.opcode.as_str(),
                "measure" | "measure_x" | "measure_y" | "measure_z"
            ) {
                for parameter in &instruction.parameters {
                    if let CircuitParameter::MeasurementIndex(index) = parameter {
                        available.insert(*index);
                    }
                }
            }
            output.push(instruction);
        }
        if output.len() > super::MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(
                "Boolean store instruction budget".to_owned(),
            ));
        }
        self.instructions = output;
        Ok(())
    }
}

fn boolean_inputs_available(
    root: CircuitScalarId,
    scalars: &[CircuitScalarNode],
    available: &HashSet<u64>,
) -> bool {
    let mut visited = HashSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        match scalars.get(id.index() as usize) {
            Some(CircuitScalarNode::Bool(_)) => {}
            Some(CircuitScalarNode::Measurement(index)) if available.contains(index) => {}
            Some(CircuitScalarNode::Apply {
                opcode,
                result_type: CircuitScalarType::Bool,
                operands,
                ..
            }) if matches!(
                opcode.as_str(),
                "classical.select"
                    | "classical.and"
                    | "classical.or"
                    | "classical.not"
                    | "classical.xor"
                    | "classical.cast"
            ) =>
            {
                pending.extend(operands.iter().copied());
            }
            _ => return false,
        }
    }
    true
}
