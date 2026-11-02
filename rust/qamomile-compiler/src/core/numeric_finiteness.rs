//! Bounded abstract interpretation for binary64 operation safety.

use thiserror::Error;

/// One scalar operand fact supplied by a verified consumer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealFinitenessOperand {
    guaranteed_finite: bool,
    static_value: Option<f64>,
}

impl RealFinitenessOperand {
    /// Construct one binary64 finiteness fact.
    #[must_use]
    pub const fn new(guaranteed_finite: bool, static_value: Option<f64>) -> Self {
        Self {
            guaranteed_finite,
            static_value,
        }
    }

    fn is_guaranteed_finite(self) -> bool {
        self.guaranteed_finite && self.static_value.is_none_or(f64::is_finite)
    }
}

/// Conservative safety facts for one `QCore` Real operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RealOperationFacts {
    runtime_guard: RealRuntimeGuard,
    guaranteed_finite: bool,
}

impl RealOperationFacts {
    const fn new(runtime_guard: RealRuntimeGuard, guaranteed_finite: bool) -> Self {
        Self {
            runtime_guard,
            guaranteed_finite,
        }
    }

    /// Return the runtime check required before evaluating the operation.
    #[must_use]
    pub const fn runtime_guard(self) -> RealRuntimeGuard {
        self.runtime_guard
    }

    /// Return whether every represented result is finite binary64.
    #[must_use]
    pub const fn guarantees_finite(self) -> bool {
        self.guaranteed_finite
    }
}

/// One target-neutral runtime check required by a Real operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealRuntimeGuard {
    /// The operation is total for every value described by its operands.
    None,
    /// The right operand must not be positive or negative zero.
    NonZeroDivisor,
    /// The operands must remain inside `QCore`'s Real power domain.
    PowerDomain,
}

impl RealRuntimeGuard {
    /// Return the stable native-projection name for this guard.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NonZeroDivisor => "nonzero_divisor",
            Self::PowerDomain => "power_domain",
        }
    }
}

/// A malformed request for the closed Real safety algebra.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum RealOperationFactsError {
    /// The operation is outside the supported safety vocabulary.
    #[error("Q4510: unsupported Real operation safety query")]
    UnsupportedOperation,
    /// The operation does not have its verified scalar arity.
    #[error("Q4511: malformed Real operation safety operand row")]
    MalformedOperands,
}

/// Derive the runtime guard and finiteness for one `QCore` Real operation.
///
/// The result is deliberately conservative. A false fact means that the
/// property may hold for a particular invocation, but is not derivable from
/// the supplied abstract operands alone.
///
/// # Errors
///
/// Returns [`RealOperationFactsError`] when the opcode is outside the closed
/// safety vocabulary or its operand row has the wrong arity.
pub fn real_operation_facts(
    opcode: &str,
    operands: &[RealFinitenessOperand],
) -> Result<RealOperationFacts, RealOperationFactsError> {
    match opcode {
        "classical.neg" | "classical.pos" => {
            let [operand] = operands else {
                return Err(RealOperationFactsError::MalformedOperands);
            };
            Ok(RealOperationFacts::new(
                RealRuntimeGuard::None,
                operand.is_guaranteed_finite(),
            ))
        }
        "classical.add" => binary_total(operands, |left, right| {
            static_binary(left, right, |left, right| left + right).unwrap_or_else(|| {
                is_static_zero(left) && right.is_guaranteed_finite()
                    || is_static_zero(right) && left.is_guaranteed_finite()
            })
        }),
        "classical.sub" => binary_total(operands, |left, right| {
            static_binary(left, right, |left, right| left - right).unwrap_or_else(|| {
                is_static_zero(left) && right.is_guaranteed_finite()
                    || is_static_zero(right) && left.is_guaranteed_finite()
            })
        }),
        "classical.mul" => binary_total(operands, |left, right| {
            static_binary(left, right, |left, right| left * right).unwrap_or_else(|| {
                preserves_finiteness_as_scale(left, right)
                    || preserves_finiteness_as_scale(right, left)
            })
        }),
        "classical.truediv" => {
            let [left, right] = operands else {
                return Err(RealOperationFactsError::MalformedOperands);
            };
            let runtime_guard = if right.static_value.is_some_and(|value| value != 0.0) {
                RealRuntimeGuard::None
            } else {
                RealRuntimeGuard::NonZeroDivisor
            };
            let guaranteed_finite = matches!(runtime_guard, RealRuntimeGuard::None)
                && static_binary(*left, *right, |left, right| left / right).unwrap_or_else(|| {
                    right.static_value.is_some_and(|right| right.abs() >= 1.0)
                        && left.is_guaranteed_finite()
                });
            Ok(RealOperationFacts::new(runtime_guard, guaranteed_finite))
        }
        "classical.pow" => {
            let [base, exponent] = operands else {
                return Err(RealOperationFactsError::MalformedOperands);
            };
            let runtime_guard = if real_power_is_total(*base, *exponent) {
                RealRuntimeGuard::None
            } else {
                RealRuntimeGuard::PowerDomain
            };
            let guaranteed_finite =
                static_binary(*base, *exponent, f64::powf).unwrap_or_else(|| {
                    exponent.static_value == Some(0.0)
                        || exponent.static_value == Some(1.0) && base.is_guaranteed_finite()
                });
            Ok(RealOperationFacts::new(runtime_guard, guaranteed_finite))
        }
        _ => Err(RealOperationFactsError::UnsupportedOperation),
    }
}

fn binary_total(
    operands: &[RealFinitenessOperand],
    finite: impl FnOnce(RealFinitenessOperand, RealFinitenessOperand) -> bool,
) -> Result<RealOperationFacts, RealOperationFactsError> {
    let [left, right] = operands else {
        return Err(RealOperationFactsError::MalformedOperands);
    };
    Ok(RealOperationFacts::new(
        RealRuntimeGuard::None,
        finite(*left, *right),
    ))
}

fn real_power_is_total(base: RealFinitenessOperand, exponent: RealFinitenessOperand) -> bool {
    if let (Some(base), Some(exponent)) = (base.static_value, exponent.static_value) {
        return !(base == 0.0 && exponent.is_finite() && exponent < 0.0
            || base < 0.0 && exponent.is_finite() && exponent.fract() != 0.0);
    }
    if base
        .static_value
        .is_some_and(|base| base.is_nan() || base > 0.0)
    {
        return true;
    }
    exponent
        .static_value
        .is_some_and(|exponent| !exponent.is_finite() || exponent >= 0.0 && exponent.fract() == 0.0)
}

fn static_binary(
    left: RealFinitenessOperand,
    right: RealFinitenessOperand,
    operation: impl FnOnce(f64, f64) -> f64,
) -> Option<bool> {
    let (Some(left), Some(right)) = (left.static_value, right.static_value) else {
        return None;
    };
    Some(operation(left, right).is_finite())
}

fn is_static_zero(operand: RealFinitenessOperand) -> bool {
    operand.static_value == Some(0.0)
}

fn preserves_finiteness_as_scale(
    scale: RealFinitenessOperand,
    value: RealFinitenessOperand,
) -> bool {
    scale
        .static_value
        .is_some_and(|scale| scale.is_finite() && scale.abs() <= 1.0)
        && value.is_guaranteed_finite()
}

#[cfg(test)]
mod tests {
    use super::{
        RealFinitenessOperand, RealOperationFacts, RealOperationFactsError, RealRuntimeGuard,
        real_operation_facts,
    };

    const FINITE: RealFinitenessOperand = RealFinitenessOperand::new(true, None);
    const UNKNOWN: RealFinitenessOperand = RealFinitenessOperand::new(false, None);

    fn literal(value: f64) -> RealFinitenessOperand {
        RealFinitenessOperand::new(value.is_finite(), Some(value))
    }

    const fn facts(guard: RealRuntimeGuard, finite: bool) -> RealOperationFacts {
        RealOperationFacts::new(guard, finite)
    }

    #[test]
    fn signs_and_zero_offsets_preserve_finiteness() {
        for opcode in ["classical.neg", "classical.pos"] {
            assert_eq!(
                real_operation_facts(opcode, &[FINITE]),
                Ok(facts(RealRuntimeGuard::None, true))
            );
            assert_eq!(
                real_operation_facts(opcode, &[UNKNOWN]),
                Ok(facts(RealRuntimeGuard::None, false))
            );
        }
        for zero in [0.0, -0.0] {
            assert_eq!(
                real_operation_facts("classical.add", &[FINITE, literal(zero)]),
                Ok(facts(RealRuntimeGuard::None, true))
            );
            assert_eq!(
                real_operation_facts("classical.sub", &[literal(zero), FINITE]),
                Ok(facts(RealRuntimeGuard::None, true))
            );
        }
        assert_eq!(
            real_operation_facts("classical.add", &[FINITE, literal(0.25)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
    }

    #[test]
    fn bounded_static_scales_preserve_finiteness() {
        for scale in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_eq!(
                real_operation_facts("classical.mul", &[FINITE, literal(scale)]),
                Ok(facts(RealRuntimeGuard::None, true))
            );
        }
        assert_eq!(
            real_operation_facts("classical.mul", &[FINITE, literal(1.000_000_1)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
        assert_eq!(
            real_operation_facts("classical.mul", &[FINITE, FINITE]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
        assert_eq!(
            real_operation_facts("classical.mul", &[UNKNOWN, literal(0.5)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
    }

    #[test]
    fn static_nonzero_divisors_separate_totality_from_finiteness() {
        assert_eq!(
            real_operation_facts("classical.truediv", &[FINITE, literal(2.0)]),
            Ok(facts(RealRuntimeGuard::None, true))
        );
        assert_eq!(
            real_operation_facts("classical.truediv", &[FINITE, literal(0.5)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
        for divisor in [0.0, -0.0] {
            assert_eq!(
                real_operation_facts("classical.truediv", &[FINITE, literal(divisor)]),
                Ok(facts(RealRuntimeGuard::NonZeroDivisor, false))
            );
        }
        assert_eq!(
            real_operation_facts("classical.truediv", &[FINITE, FINITE]),
            Ok(facts(RealRuntimeGuard::NonZeroDivisor, false))
        );
        for divisor in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                real_operation_facts("classical.truediv", &[FINITE, literal(divisor)]),
                Ok(facts(RealRuntimeGuard::None, true))
            );
        }
        assert_eq!(
            real_operation_facts("classical.truediv", &[FINITE, literal(f64::NAN)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
    }

    #[test]
    fn closed_literals_use_binary64_result_finiteness() {
        assert_eq!(
            real_operation_facts("classical.mul", &[literal(f64::MAX), literal(2.0)]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
        assert_eq!(
            real_operation_facts("classical.sub", &[literal(f64::MAX), literal(f64::MAX)]),
            Ok(facts(RealRuntimeGuard::None, true))
        );
    }

    #[test]
    fn real_power_separates_domain_checks_from_result_finiteness() {
        for exponent in [0.0, 2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                real_operation_facts("classical.pow", &[UNKNOWN, literal(exponent)]),
                Ok(facts(RealRuntimeGuard::None, exponent == 0.0,))
            );
        }
        assert_eq!(
            real_operation_facts("classical.pow", &[FINITE, literal(1.0)]),
            Ok(facts(RealRuntimeGuard::None, true))
        );
        for exponent in [-1.0, 0.5, -0.5] {
            assert_eq!(
                real_operation_facts("classical.pow", &[UNKNOWN, literal(exponent)]),
                Ok(facts(RealRuntimeGuard::PowerDomain, false))
            );
        }
        assert_eq!(
            real_operation_facts("classical.pow", &[literal(2.0), UNKNOWN]),
            Ok(facts(RealRuntimeGuard::None, false))
        );
        assert_eq!(
            real_operation_facts("classical.pow", &[literal(-2.0), UNKNOWN]),
            Ok(facts(RealRuntimeGuard::PowerDomain, false))
        );
        assert_eq!(
            real_operation_facts("classical.pow", &[literal(-2.0), literal(0.5)]),
            Ok(facts(RealRuntimeGuard::PowerDomain, false))
        );
    }

    #[test]
    fn unknown_operations_and_wrong_arities_fail_closed() {
        assert_eq!(
            real_operation_facts("classical.sqrt", &[FINITE]),
            Err(RealOperationFactsError::UnsupportedOperation)
        );
        assert_eq!(
            real_operation_facts("classical.neg", &[]),
            Err(RealOperationFactsError::MalformedOperands)
        );
        assert_eq!(
            real_operation_facts("classical.add", &[FINITE]),
            Err(RealOperationFactsError::MalformedOperands)
        );
    }
}
