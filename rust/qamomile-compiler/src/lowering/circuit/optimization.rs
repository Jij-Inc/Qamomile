//! Exact local circuit rewriting with source-check and segment boundaries.

use std::collections::{HashMap, HashSet};

use super::{
    CircuitDefinition, CircuitParameter, StaticCircuit, StaticCircuitError, StaticInstruction,
    instruction_count,
};

impl StaticCircuit {
    /// Simplify projected gates without changing phase, wires, or measurement slots.
    ///
    /// # Errors
    /// Rejects excessive work or region depth before returning a candidate.
    pub fn simplify_exact(mut self, max_work: u64) -> Result<Self, StaticCircuitError> {
        let count =
            u64::try_from(self.operation_count).map_err(|_| StaticCircuitError::Overflow)?;
        if max_work == 0 || count > max_work {
            return Err(StaticCircuitError::Unsupported(
                "circuit optimization work budget exceeded".into(),
            ));
        }
        let barriers = self
            .closed_cuts
            .iter()
            .map(|c| c.instruction_index)
            .chain(self.scalar_checks.iter().map(|c| c.instruction_index))
            .collect::<HashSet<_>>();
        let mut work = RewriteWork::new(max_work);
        let (instructions, offsets) =
            simplify(self.instructions.into_vec(), &barriers, 0, &mut work)?;
        for cut in &mut self.closed_cuts {
            cut.instruction_index = offsets[cut.instruction_index];
        }
        for check in &mut self.scalar_checks {
            check.instruction_index = offsets[check.instruction_index];
        }
        self.operation_count = instruction_count(&instructions);
        self.instructions = instructions.into_boxed_slice();
        self.optimization_stages
            .push("optimization:circuit.exact-local@1".into());
        Ok(self)
    }

    /// Return late passes actually executed on this projection.
    #[must_use]
    pub fn optimization_stages(&self) -> &[String] {
        &self.optimization_stages
    }

    pub(crate) fn inherit_optimization_stages(mut self, source: &Self) -> Self {
        self.optimization_stages
            .clone_from(&source.optimization_stages);
        self
    }

    pub(crate) fn record_realization_selection(&mut self) {
        self.optimization_stages
            .push("optimization:realization.minimum-selectors@1".into());
    }
}

struct RewriteWork {
    remaining: u64,
    // Retain both owners so pointer identities cannot be reused during rewriting.
    definitions: HashMap<usize, (CircuitDefinition, CircuitDefinition)>,
}

impl RewriteWork {
    fn new(remaining: u64) -> Self {
        Self {
            remaining,
            definitions: HashMap::new(),
        }
    }

    fn definition(
        &mut self,
        source: &CircuitDefinition,
        depth: usize,
    ) -> Result<CircuitDefinition, StaticCircuitError> {
        if let Some((_, result)) = self.definitions.get(&source.identity()) {
            return Ok(result.clone());
        }
        // Do not force symbolic arithmetic recipes to allocate gates early.
        let Some(body) = source.retained_body() else {
            return Ok(source.clone());
        };
        if body.len() as u64 > self.remaining {
            return Err(StaticCircuitError::Unsupported(
                "circuit optimization work budget exceeded".into(),
            ));
        }
        let (instructions, _) = simplify(body.to_vec(), &HashSet::new(), depth + 1, self)?;
        let result = CircuitDefinition::body(source.width(), instructions, source.scalars().into());
        self.definitions
            .insert(source.identity(), (source.clone(), result.clone()));
        Ok(result)
    }
}

fn simplify(
    input: Vec<StaticInstruction>,
    barriers: &HashSet<usize>,
    depth: usize,
    work: &mut RewriteWork,
) -> Result<(Vec<StaticInstruction>, Vec<usize>), StaticCircuitError> {
    if depth >= 256 {
        return Err(StaticCircuitError::Unsupported(
            "circuit optimization region depth exceeded".into(),
        ));
    }
    work.remaining = work
        .remaining
        .checked_sub(input.len() as u64)
        .ok_or_else(|| {
            StaticCircuitError::Unsupported("circuit optimization work budget exceeded".into())
        })?;
    let mut output: Vec<StaticInstruction> = Vec::with_capacity(input.len());
    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut floor = 0;
    for (index, mut instruction) in input.into_iter().enumerate() {
        if barriers.contains(&index) {
            floor = output.len();
        }
        offsets.push(output.len());
        for block in &mut instruction.regions {
            let (nested, _) = simplify(
                std::mem::take(&mut block.instructions).into_vec(),
                &HashSet::new(),
                depth + 1,
                work,
            )?;
            block.instructions = nested.into_boxed_slice();
        }
        for parameter in &mut instruction.parameters {
            if let CircuitParameter::Definition(definition) = parameter {
                *definition = work.definition(definition, depth)?;
            }
        }
        if !reduce_tail(&mut output, &mut instruction, floor) {
            continue;
        }
        output.push(instruction);
    }
    offsets.push(output.len());
    Ok((output, offsets))
}

/// Every retry contracts the stack, so reaching a local fixed point is linear
/// in the input length and cannot move a rewrite past a source-check barrier.
fn reduce_tail(
    output: &mut Vec<StaticInstruction>,
    instruction: &mut StaticInstruction,
    floor: usize,
) -> bool {
    loop {
        if output.len() > floor && compatible(&output[output.len() - 1], instruction) {
            let last = &output[output.len() - 1];
            if inverse(&last.opcode) == Some(instruction.opcode.as_str()) {
                output.pop();
                return false;
            }
            if last.opcode == "reset" && instruction.opcode == "reset" {
                return false;
            }
            let fused = match (last.opcode.as_str(), instruction.opcode.as_str()) {
                ("t", "t") => Some("s"),
                ("tdg", "tdg") => Some("sdg"),
                ("s", "s") | ("sdg", "sdg") => Some("z"),
                _ => None,
            };
            if let Some(opcode) = fused {
                output.pop();
                instruction.opcode = opcode.into();
                continue;
            }
        }
        if instruction.opcode == "h" && output.len() >= floor + 2 {
            let a = &output[output.len() - 2];
            let b = &output[output.len() - 1];
            if a.opcode == "h" && compatible(a, instruction) && compatible(b, instruction) {
                let replacement = match b.opcode.as_str() {
                    "x" => Some("z"),
                    "z" => Some("x"),
                    _ => None,
                };
                if let Some(opcode) = replacement {
                    output.truncate(output.len() - 2);
                    instruction.opcode = opcode.into();
                    continue;
                }
            }
        }
        return true;
    }
}

fn compatible(left: &StaticInstruction, right: &StaticInstruction) -> bool {
    left.qubits == right.qubits
        && left.controls == right.controls
        && left.condition == right.condition
        && left.parameters.is_empty()
        && right.parameters.is_empty()
        && left.regions.is_empty()
        && right.regions.is_empty()
}

fn inverse(opcode: &str) -> Option<&'static str> {
    match opcode {
        "h" => Some("h"),
        "x" => Some("x"),
        "y" => Some("y"),
        "z" => Some("z"),
        "cx" => Some("cx"),
        "cy" => Some("cy"),
        "cz" => Some("cz"),
        "ch" => Some("ch"),
        "ccx" => Some("ccx"),
        "swap" => Some("swap"),
        "s" => Some("sdg"),
        "sdg" => Some("s"),
        "t" => Some("tdg"),
        "tdg" => Some("t"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(op: &str) -> StaticInstruction {
        StaticInstruction::test_new(op, vec![0], vec![], vec![], None, vec![])
    }

    #[test]
    fn exact_phase_rules_and_barriers() {
        let (result, _) = simplify(
            vec![gate("h"), gate("z"), gate("h"), gate("t"), gate("t")],
            &HashSet::new(),
            0,
            &mut RewriteWork::new(100),
        )
        .unwrap();
        assert_eq!(
            result
                .iter()
                .map(StaticInstruction::opcode)
                .collect::<Vec<_>>(),
            ["x", "s"]
        );
        let (result, offsets) = simplify(
            vec![gate("h"), gate("h")],
            &HashSet::from([1]),
            0,
            &mut RewriteWork::new(100),
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(offsets, [0, 1, 2]);
        let (result, _) = simplify(
            vec![gate("s"), gate("t"), gate("t"), gate("z")],
            &HashSet::new(),
            0,
            &mut RewriteWork::new(4),
        )
        .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn shared_bodies_are_rewritten_once_with_a_shared_budget() {
        let source =
            CircuitDefinition::body(1, vec![gate("h"), gate("z"), gate("h")], Box::new([]));
        let call = StaticInstruction::test_new(
            "box:helper",
            vec![0],
            vec![CircuitParameter::Definition(source.clone())],
            vec![],
            None,
            vec![],
        );
        let mut work = RewriteWork::new(5);
        let (result, _) = simplify(
            vec![call.clone(), call.clone()],
            &HashSet::new(),
            0,
            &mut work,
        )
        .unwrap();
        let CircuitParameter::Definition(first) = &result[0].parameters[0] else {
            panic!("definition");
        };
        let CircuitParameter::Definition(second) = &result[1].parameters[0] else {
            panic!("definition");
        };
        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.instructions().unwrap()[0].opcode(), "x");
        assert_eq!(source.instructions().unwrap().len(), 3);
        assert_eq!(work.remaining, 0);
        assert!(
            simplify(
                vec![call.clone(), call],
                &HashSet::new(),
                0,
                &mut RewriteWork::new(4)
            )
            .is_err()
        );
    }
}
