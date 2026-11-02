//! Rejection boundary for function-valued HIR results.
//!
//! Qamomile permits closed function values as direct static call arguments.
//! It does not expose qkernels returning qkernels. Keeping that distinction
//! here makes the native HIR boundary fail closed without retaining a second
//! defunctionalization pipeline for function results.

use super::{HirLoweringError, HirNormalizationWork, wire_type_is_function};
use crate::{types::TypeArena, wire};

/// Reject function-valued declaration results at the native trust boundary.
pub(super) fn lower(
    functions: &mut [wire::HirFunctionProto],
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    for function in functions {
        reject_function_result(function, work)?;
    }
    Ok(())
}

/// Reject a generated specialization that acquired a function-valued result.
pub(super) fn lower_generated(
    function: &mut wire::HirFunctionProto,
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    reject_function_result(function, work)
}

fn reject_function_result(
    function: &wire::HirFunctionProto,
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    work.visit_function_result();
    let result_type = function.result_type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore(format!("function {:?} has no result type", function.symbol))
    })?;
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope(&function.symbol);
    if wire_type_is_function(result_type, &scope, &mut arena)? {
        return Err(HirLoweringError::Unsupported(format!(
            "function {:?} has a function-valued result; pass closed functions as direct arguments",
            function.symbol
        )));
    }
    Ok(())
}
