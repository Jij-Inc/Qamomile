//! Dominance-local common expressions without evaluating symbolic values.

use std::collections::HashMap;

use prost::Message;

use super::optimization_pipeline::preflight_work;
use crate::operations::ClassicalOpcode;
use crate::{OptimizationError, VerifiedModule, VerifiedProgram, seal_package, wire};

/// Verified output and accounting from one common-expression pass.
#[derive(Clone, Debug)]
pub struct ClassicalCseResult<P> {
    /// The transformed module or selected program, reverified after rewriting.
    pub program: P,
    /// Work and changes made by this invocation.
    pub work: ClassicalCseWork,
}

/// Work and changes from a bounded classical common-expression traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassicalCseWork {
    visits: u64,
    removed: u64,
}

impl ClassicalCseWork {
    /// Return the number of visited retained operations.
    #[must_use]
    pub const fn visits(self) -> u64 {
        self.visits
    }

    /// Return the number of redundant definitions removed.
    #[must_use]
    pub const fn removed(self) -> u64 {
        self.removed
    }
}

/// Rewrite one retained package and reverify it through the owner's decoder.
///
/// Module and program share the traversal; only the verified typestate differs.
fn run<P>(
    package: &wire::QCorePackageProto,
    max_work: u64,
    reverify: impl FnOnce(&[u8]) -> Result<P, crate::CompilerError>,
) -> Result<ClassicalCseResult<P>, OptimizationError> {
    preflight_work(package, max_work, 1)?;
    let mut package = package.clone();
    let mut work = ClassicalCseWork::default();
    for function in &mut package.functions {
        if let Some(body) = &mut function.body {
            rewrite_block(body, &mut HashMap::new(), &mut work);
        }
    }
    let bytes = seal_package(package).map_err(|e| OptimizationError::invalid(e.to_string()))?;
    let program = reverify(&bytes).map_err(|e| OptimizationError::invalid(e.to_string()))?;
    Ok(ClassicalCseResult { program, work })
}

fn rewrite_block(
    block: &mut wire::BlockProto,
    replacements: &mut HashMap<u64, u64>,
    work: &mut ClassicalCseWork,
) {
    // Tables are block-local: no speculation, sibling lookup, or quadratic table
    // copying at region entry. SSA IDs are unique within a verified function.
    let mut expressions: HashMap<Vec<u8>, u64> = HashMap::new();
    block.operations.retain_mut(|operation| {
        work.visits += 1;
        for operand in &mut operation.operands {
            if let Some(value) = replacements.get(operand) {
                *operand = *value;
            }
        }
        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                rewrite_block(nested, replacements, work);
            }
        }
        if operation.results.len() != 1
            || operation.results[0].has_region
            || !operation.regions.is_empty()
        {
            return true;
        }
        // Classical primitives are deterministic functions of immutable inputs.
        // They may fail: retaining the first dominating evaluation preserves
        // that failure. Calls and quantum operations are deliberately excluded.
        if ClassicalOpcode::parse(&operation.opcode).is_none()
            && !matches!(
                operation.opcode.as_str(),
                "qcore.constant" | "qcore.stage_lift"
            )
        {
            return true;
        }
        let mut key = operation.clone();
        key.span = None;
        key.results[0].id = 0;
        key.results[0].hint.clear();
        let key = key.encode_to_vec();
        if let Some(previous) = expressions.get(&key) {
            replacements.insert(operation.results[0].id, *previous);
            work.removed += 1;
            false
        } else {
            expressions.insert(key, operation.results[0].id);
            true
        }
    });
    if let Some(terminator) = &mut block.terminator {
        for operand in &mut terminator.operands {
            if let Some(value) = replacements.get(operand) {
                *operand = *value;
            }
        }
    }
}

impl VerifiedModule {
    /// Require every original callable declaration at a symbolic pass boundary.
    ///
    /// # Errors
    /// Rejects removed or changed public names, types, defaults, or result types.
    pub fn validate_optimization_result(&self, result: &Self) -> Result<(), OptimizationError> {
        let source = &self.local.checked.package;
        let candidate = &result.local.checked.package;
        if source.module_name != candidate.module_name
            || source.functions.len() != candidate.functions.len()
        {
            return Err(OptimizationError::invalid(
                "symbolic pass changed module interface",
            ));
        }
        let functions = candidate
            .functions
            .iter()
            .map(|f| (f.symbol.as_str(), f))
            .collect::<HashMap<_, _>>();
        for function in &source.functions {
            let other = functions
                .get(function.symbol.as_str())
                .ok_or_else(|| OptimizationError::invalid("symbolic pass removed callable"))?;
            if function.result_type != other.result_type
                || function.parameters.len() != other.parameters.len()
                || function.external != other.external
                || !function
                    .parameters
                    .iter()
                    .zip(&other.parameters)
                    .all(|(a, b)| {
                        a.name == b.name
                            && a.has_default == b.has_default
                            && a.default_value == b.default_value
                            && a.value.as_ref().map(|v| &v.r#type)
                                == b.value.as_ref().map(|v| &v.r#type)
                    })
            {
                return Err(OptimizationError::invalid(
                    "symbolic pass changed callable interface",
                ));
            }
        }
        Ok(())
    }

    /// Share repeated classical expressions while retaining symbolic families.
    ///
    /// # Errors
    /// Rejects excessive work, depth, or an invalid transformed module.
    pub fn eliminate_common_expressions(
        &self,
        max_work: u64,
    ) -> Result<ClassicalCseResult<Self>, OptimizationError> {
        run(&self.local.checked.package, max_work, Self::from_bytes)
    }
}

impl VerifiedProgram {
    /// Share dominated identical classical operations without reassociation.
    ///
    /// # Errors
    /// Rejects excessive work, depth, or invalid ownership/cleanup after rewriting.
    pub fn eliminate_common_expressions(
        &self,
        max_work: u64,
    ) -> Result<ClassicalCseResult<Self>, OptimizationError> {
        run(&self.local.checked.package, max_work, Self::from_bytes)
    }
}
