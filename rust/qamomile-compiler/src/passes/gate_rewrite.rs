//! Bounded extension-defined gate rewrites over verified `QCore`.

use super::optimization_pipeline::preflight_work;
use crate::operations::{GateOpcode, primitive_gate_spec};
use crate::{OptimizationError, VerifiedProgram, seal_package, wire};

/// One bounded contracting rewrite over consecutive identical-selector unary gates.
///
/// The author asserts equivalence, including phase, on the defined execution domain.
#[derive(Clone, Debug)]
pub struct GateRewriteRule {
    pattern: Vec<GateOpcode>,
    replacement: Vec<GateOpcode>,
}

impl GateRewriteRule {
    /// Check a non-expanding unary, angle-free gate rule.
    ///
    /// # Errors
    /// Rejects empty, expanding, oversized, or non-unary gate sequences.
    pub fn new(pattern: Vec<String>, replacement: Vec<String>) -> Result<Self, OptimizationError> {
        if pattern.is_empty()
            || pattern.len() > 16
            || replacement.is_empty()
            || replacement.len() > pattern.len()
            || pattern.iter().chain(&replacement).any(|name| {
                primitive_gate_spec(name)
                    .is_none_or(|spec| spec.has_angle() || spec.selectors().len() != 1)
            })
        {
            return Err(OptimizationError::invalid(
                "rewrite requires 1..16 angle-free unary gates and a nonempty contracting replacement",
            ));
        }
        let parse = |names: Vec<String>| {
            names
                .into_iter()
                .map(|name| {
                    GateOpcode::parse(&name)
                        .ok_or_else(|| OptimizationError::invalid("unknown primitive gate"))
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            pattern: parse(pattern)?,
            replacement: parse(replacement)?,
        })
    }

    /// Apply one rule once per source position using shared native SSA bookkeeping.
    ///
    /// # Errors
    /// Rejects depth/work exhaustion or a result rejected by complete verification.
    pub fn apply(
        &self,
        program: &VerifiedProgram,
        max_work: u64,
    ) -> Result<VerifiedProgram, OptimizationError> {
        preflight_work(&program.local.checked.package, max_work, 1)?;
        let mut package = program.local.checked.package.clone();
        let mut remaining = max_work;
        for function in &mut package.functions {
            if let Some(body) = &mut function.body {
                self.rewrite_block(body, 0, &mut remaining)?;
            }
        }
        let bytes =
            seal_package(package).map_err(|error| OptimizationError::invalid(error.to_string()))?;
        VerifiedProgram::from_bytes(&bytes)
            .map_err(|error| OptimizationError::invalid(error.to_string()))
    }

    fn rewrite_block(
        &self,
        block: &mut wire::BlockProto,
        depth: usize,
        work: &mut u64,
    ) -> Result<(), OptimizationError> {
        if depth >= 256 {
            return Err(OptimizationError::invalid("rewrite depth exceeds 256"));
        }
        let source = std::mem::take(&mut block.operations);
        let mut output = Vec::new();
        for mut operation in source {
            *work = work
                .checked_sub(1)
                .ok_or_else(|| OptimizationError::invalid("rewrite work budget exceeded"))?;
            for region in &mut operation.regions {
                for nested in &mut region.blocks {
                    self.rewrite_block(nested, depth + 1, work)?;
                }
            }
            output.push(operation);
            if output.len() >= self.pattern.len() {
                let start = output.len() - self.pattern.len();
                let matched = &output[start..];
                if matched.iter().zip(&self.pattern).all(|(op, name)| {
                    op.opcode == name.as_str()
                        && op.results.len() == 1
                        && op.operands.len() == matched[0].operands.len()
                        && op.operands.get(1..) == matched[0].operands.get(1..)
                        && op.attributes == matched[0].attributes
                }) && matched
                    .windows(2)
                    .all(|pair| pair[0].results[0].id == pair[1].operands[0])
                {
                    let last_result = matched.last().expect("nonempty pattern").results.clone();
                    let mut replacement = matched[..self.replacement.len()].to_vec();
                    for (op, name) in replacement.iter_mut().zip(&self.replacement) {
                        name.as_str().clone_into(&mut op.opcode);
                        // A replacement is a new operation: it carries no intent
                        // from the gates it supersedes.
                        op.intent.clear();
                    }
                    replacement
                        .last_mut()
                        .expect("nonempty replacement")
                        .results = last_result;
                    output.truncate(start);
                    output.extend(replacement);
                }
            }
        }
        block.operations = output;
        Ok(())
    }
}
