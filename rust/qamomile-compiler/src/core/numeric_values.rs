//! Closed Python-compatible numeric algebra for native static evaluation.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::{FromPrimitive, Signed, ToPrimitive, Zero};

use crate::static_values::{StaticScalar, StaticValue};
use crate::wire;

const MAX_INTEGER_RESULT_BITS: u64 = 8 * 1024 * 1024;
const INTEGER_LOG2_RETAINED_BITS: u64 = 54;
const INTEGER_LOG2_MANTISSA_BITS: i32 = 53;

/// A selected exact numeric expression that is undefined or too large to seal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NumericEvaluationError {
    /// Python defines this operand combination as an exception.
    Undefined(&'static str),
    /// The exact result cannot fit within the persistent artifact budget.
    Limit(&'static str),
}

#[derive(Clone, Debug)]
enum Number {
    Int(BigInt),
    Real(f64),
    Complex(Complex64),
}

impl Number {
    fn from_static(value: &StaticValue) -> Option<Self> {
        match value {
            StaticValue::Scalar(StaticScalar::Int(value)) => Some(Self::Int(BigInt::from(*value))),
            StaticValue::Scalar(StaticScalar::RealBits(value)) => {
                Some(Self::Real(f64::from_bits(*value)))
            }
            StaticValue::BigInt(value) => Some(Self::Int(BigInt::from_signed_bytes_be(value))),
            StaticValue::ComplexBits(real, imaginary) => Some(Self::Complex(Complex64::new(
                f64::from_bits(*real),
                f64::from_bits(*imaginary),
            ))),
            StaticValue::Unit
            | StaticValue::Scalar(StaticScalar::Bool(_))
            | StaticValue::String(_)
            | StaticValue::Bytes(_)
            | StaticValue::Aggregate(_)
            | StaticValue::Record(_)
            | StaticValue::Mapping(_) => None,
        }
    }

    const fn rank(&self) -> u8 {
        match self {
            Self::Int(_) => 0,
            Self::Real(_) => 1,
            Self::Complex(_) => 2,
        }
    }

    fn into_static(self) -> StaticValue {
        match self {
            Self::Int(value) => integer_static(&value),
            Self::Real(value) => StaticValue::Scalar(StaticScalar::RealBits(value.to_bits())),
            Self::Complex(value) => {
                StaticValue::ComplexBits(value.re.to_bits(), value.im.to_bits())
            }
        }
    }

    fn into_real(self) -> Result<f64, NumericEvaluationError> {
        match self {
            Self::Int(value) => integer_to_real(&value),
            Self::Real(value) => Ok(value),
            Self::Complex(_) => Err(NumericEvaluationError::Undefined(
                "Complex cannot be converted to Real implicitly",
            )),
        }
    }

    fn into_complex(self) -> Result<Complex64, NumericEvaluationError> {
        match self {
            Self::Int(value) => Ok(Complex64::new(integer_to_real(&value)?, 0.0)),
            Self::Real(value) => Ok(Complex64::new(value, 0.0)),
            Self::Complex(value) => Ok(value),
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Int(value) => value.is_zero(),
            Self::Real(value) => *value == 0.0,
            Self::Complex(value) => value.re == 0.0 && value.im == 0.0,
        }
    }
}

fn integer_to_real(value: &BigInt) -> Result<f64, NumericEvaluationError> {
    let converted = value.to_f64().filter(|value| value.is_finite());
    converted.ok_or(NumericEvaluationError::Undefined(
        "integer is too large to convert to binary64",
    ))
}

/// Evaluate one operation from `QCore`'s closed numeric vocabulary.
pub(crate) fn evaluate(
    operation: &wire::OperationProto,
    operands: &[&StaticValue],
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    if operation.opcode == "classical.cast" {
        if !supports(operation.opcode.as_str(), operands) {
            return Ok(None);
        }
        return cast(operation, operands).map(Some);
    }
    evaluate_plain(operation.opcode.as_str(), operands)
}

/// Evaluate one metadata-free operation from `QCore`'s numeric vocabulary.
pub(crate) fn evaluate_plain(
    opcode: &str,
    operands: &[&StaticValue],
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    if !supports(opcode, operands) {
        return Ok(None);
    }
    let value = match opcode {
        "classical.add" => binary_numeric(operands, BinaryNumeric::Add)?,
        "classical.sub" => binary_numeric(operands, BinaryNumeric::Sub)?,
        "classical.mul" => binary_numeric(operands, BinaryNumeric::Mul)?,
        "classical.truediv" => true_division(operands)?,
        "classical.floordiv" => floor_or_mod(operands, false)?,
        "classical.mod" => floor_or_mod(operands, true)?,
        "classical.pow" => power(operands)?,
        "classical.and_bits" => integer_binary(operands, IntegerBinary::And)?,
        "classical.or_bits" => integer_binary(operands, IntegerBinary::Or)?,
        "classical.xor" => integer_binary(operands, IntegerBinary::Xor)?,
        "classical.lshift" => shift(operands, true)?,
        "classical.rshift" => shift(operands, false)?,
        "classical.invert" => unary_integer(operands, |value| !value)?,
        "classical.neg" => unary_numeric(operands, true)?,
        "classical.pos" => unary_numeric(operands, false)?,
        "classical.ceil" => ceiling(operands)?,
        "classical.log2" => base_two_logarithm(operands)?,
        "classical.sqrt" => square_root(operands)?,
        _ => return Ok(None),
    };
    Ok(Some(value))
}

fn supports(opcode: &str, operands: &[&StaticValue]) -> bool {
    match opcode {
        "classical.add" | "classical.sub" | "classical.mul" | "classical.truediv"
        | "classical.floordiv" | "classical.mod" | "classical.pow" | "classical.neg"
        | "classical.pos" | "classical.cast" | "classical.ceil" | "classical.log2"
        | "classical.sqrt" => operands
            .iter()
            .all(|value| Number::from_static(value).is_some()),
        "classical.and_bits" | "classical.or_bits" | "classical.xor" | "classical.lshift"
        | "classical.rshift" | "classical.invert" => {
            operands.iter().all(|value| integer_value(value).is_some())
        }
        _ => false,
    }
}

fn ceiling(operands: &[&StaticValue]) -> Result<StaticValue, NumericEvaluationError> {
    let operand = unary_number(operands)?;
    match operand {
        Number::Int(value) => checked_static(Number::Int(value)),
        Number::Real(value) if value.is_finite() => {
            let value = BigInt::from_f64(value.ceil()).ok_or(NumericEvaluationError::Undefined(
                "ceil result cannot be represented as Int",
            ))?;
            checked_static(Number::Int(value))
        }
        Number::Real(_) => Err(NumericEvaluationError::Undefined(
            "ceil input must be finite",
        )),
        Number::Complex(_) => Err(NumericEvaluationError::Undefined(
            "ceil is undefined for Complex",
        )),
    }
}

fn base_two_logarithm(operands: &[&StaticValue]) -> Result<StaticValue, NumericEvaluationError> {
    let operand = unary_number(operands)?;
    let value = match operand {
        Number::Int(value) if value.is_positive() => integer_log2(&value)?,
        Number::Real(value) if value.is_finite() && value > 0.0 => value.log2(),
        Number::Int(_) | Number::Real(_) => {
            return Err(NumericEvaluationError::Undefined(
                "log2 input must be finite and strictly positive",
            ));
        }
        Number::Complex(_) => {
            return Err(NumericEvaluationError::Undefined(
                "log2 is undefined for Complex",
            ));
        }
    };
    Ok(Number::Real(value).into_static())
}

fn square_root(operands: &[&StaticValue]) -> Result<StaticValue, NumericEvaluationError> {
    let operand = unary_number(operands)?.into_real()?;
    if !operand.is_finite() || operand < 0.0 {
        return Err(NumericEvaluationError::Undefined(
            "sqrt input must be finite and nonnegative",
        ));
    }
    Ok(Number::Real(operand.sqrt()).into_static())
}

fn integer_log2(value: &BigInt) -> Result<f64, NumericEvaluationError> {
    if let Some(value) = value.to_f64().filter(|value| value.is_finite()) {
        return Ok(value.log2());
    }
    let bits = value.bits();
    let shift = usize::try_from(bits.saturating_sub(INTEGER_LOG2_RETAINED_BITS)).map_err(|_| {
        NumericEvaluationError::Limit("integer log2 exceeds the static evaluation budget")
    })?;
    let leading = (value >> shift)
        .to_f64()
        .ok_or(NumericEvaluationError::Undefined(
            "integer log2 leading bits cannot be converted to binary64",
        ))?;
    let normalized = leading / 2.0_f64.powi(INTEGER_LOG2_MANTISSA_BITS);
    let exponent = (bits - 1).to_f64().ok_or(NumericEvaluationError::Limit(
        "integer log2 exponent exceeds binary64",
    ))?;
    Ok(exponent + normalized.log2())
}

fn unary_number(operands: &[&StaticValue]) -> Result<Number, NumericEvaluationError> {
    let [operand] = operands else {
        return Err(NumericEvaluationError::Undefined(
            "malformed unary numeric operation",
        ));
    };
    Number::from_static(operand).ok_or(NumericEvaluationError::Undefined(
        "numeric operation received a non-numeric value",
    ))
}

/// Compare two exact numeric values using Python equality semantics.
#[allow(clippy::float_cmp)]
pub(crate) fn equal(left: &StaticValue, right: &StaticValue) -> Option<bool> {
    let left = Number::from_static(left)?;
    let right = Number::from_static(right)?;
    match (left, right) {
        (Number::Int(left), Number::Int(right)) => Some(left == right),
        (Number::Real(left), Number::Real(right)) => Some(left == right),
        (Number::Int(left), Number::Real(right)) => {
            Some(compare_integer_real(&left, right)? == Ordering::Equal)
        }
        (Number::Real(left), Number::Int(right)) => {
            Some(compare_integer_real(&right, left)? == Ordering::Equal)
        }
        (Number::Complex(left), Number::Complex(right)) => Some(left == right),
        (Number::Complex(left), Number::Int(right)) => {
            Some(left.im == 0.0 && compare_integer_real(&right, left.re)? == Ordering::Equal)
        }
        (Number::Int(left), Number::Complex(right)) => {
            Some(right.im == 0.0 && compare_integer_real(&left, right.re)? == Ordering::Equal)
        }
        (Number::Complex(left), Number::Real(right)) => Some(left.im == 0.0 && left.re == right),
        (Number::Real(left), Number::Complex(right)) => Some(right.im == 0.0 && left == right.re),
    }
}

/// Order two exact non-complex numeric values using Python semantics.
pub(crate) fn order(left: &StaticValue, right: &StaticValue) -> Option<Ordering> {
    let left = Number::from_static(left)?;
    let right = Number::from_static(right)?;
    match (left, right) {
        (Number::Int(left), Number::Int(right)) => Some(left.cmp(&right)),
        (Number::Real(left), Number::Real(right)) => left.partial_cmp(&right),
        (Number::Int(left), Number::Real(right)) => compare_integer_real(&left, right),
        (Number::Real(left), Number::Int(right)) => {
            compare_integer_real(&right, left).map(Ordering::reverse)
        }
        (Number::Complex(_), _) | (_, Number::Complex(_)) => None,
    }
}

fn compare_integer_real(integer: &BigInt, real: f64) -> Option<Ordering> {
    if real.is_nan() {
        return None;
    }
    if real == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if real == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    let truncated = BigInt::from_f64(real.trunc())?;
    match integer.cmp(&truncated) {
        Ordering::Equal if real.fract() > 0.0 => Some(Ordering::Less),
        Ordering::Equal if real.fract() < 0.0 => Some(Ordering::Greater),
        ordering => Some(ordering),
    }
}

#[derive(Clone, Copy)]
enum BinaryNumeric {
    Add,
    Sub,
    Mul,
}

fn binary_numeric(
    operands: &[&StaticValue],
    operation: BinaryNumeric,
) -> Result<StaticValue, NumericEvaluationError> {
    let [left, right] = numeric_operands(operands)?;
    let rank = left.rank().max(right.rank());
    let value = match rank {
        0 => match (left, right) {
            (Number::Int(left), Number::Int(right)) => Number::Int(match operation {
                BinaryNumeric::Add => left + right,
                BinaryNumeric::Sub => left - right,
                BinaryNumeric::Mul => left * right,
            }),
            _ => unreachable!("rank zero contains integers only"),
        },
        1 => {
            let left = left.into_real()?;
            let right = right.into_real()?;
            Number::Real(match operation {
                BinaryNumeric::Add => left + right,
                BinaryNumeric::Sub => left - right,
                BinaryNumeric::Mul => left * right,
            })
        }
        _ => {
            let left = left.into_complex()?;
            let right = right.into_complex()?;
            Number::Complex(match operation {
                BinaryNumeric::Add => left + right,
                BinaryNumeric::Sub => left - right,
                BinaryNumeric::Mul => left * right,
            })
        }
    };
    checked_static(value)
}

fn true_division(operands: &[&StaticValue]) -> Result<StaticValue, NumericEvaluationError> {
    let [left, right] = numeric_operands(operands)?;
    if right.is_zero() {
        return Err(NumericEvaluationError::Undefined("division by zero"));
    }
    if left.rank().max(right.rank()) == 2 {
        Ok(Number::Complex(left.into_complex()? / right.into_complex()?).into_static())
    } else {
        Ok(Number::Real(left.into_real()? / right.into_real()?).into_static())
    }
}

fn floor_or_mod(
    operands: &[&StaticValue],
    remainder: bool,
) -> Result<StaticValue, NumericEvaluationError> {
    let [left, right] = numeric_operands(operands)?;
    if right.is_zero() {
        return Err(NumericEvaluationError::Undefined("division by zero"));
    }
    if left.rank().max(right.rank()) == 0 {
        let (Number::Int(left), Number::Int(right)) = (left, right) else {
            unreachable!("rank zero contains integers only");
        };
        let (quotient, modulo) = python_integer_divmod(left, right);
        return checked_static(Number::Int(if remainder { modulo } else { quotient }));
    }
    let (quotient, modulo) = python_real_divmod(left.into_real()?, right.into_real()?);
    Ok(Number::Real(if remainder { modulo } else { quotient }).into_static())
}

fn power(operands: &[&StaticValue]) -> Result<StaticValue, NumericEvaluationError> {
    let [base, exponent] = numeric_operands(operands)?;
    let rank = base.rank().max(exponent.rank());
    match rank {
        0 => {
            let (Number::Int(base), Number::Int(exponent)) = (base, exponent) else {
                unreachable!("rank zero contains integers only");
            };
            if exponent.is_negative() {
                return Err(NumericEvaluationError::Undefined(
                    "negative integer exponent is outside QCore Int power",
                ));
            }
            if exponent.is_zero() {
                return Ok(integer_static(&BigInt::from(1)));
            }
            if base.is_zero() {
                return Ok(integer_static(&BigInt::ZERO));
            }
            if base == BigInt::from(1) {
                return Ok(integer_static(&base));
            }
            if base == BigInt::from(-1) {
                let result = if (&exponent & BigInt::from(1)).is_zero() {
                    BigInt::from(1)
                } else {
                    base
                };
                return Ok(integer_static(&result));
            }
            let exponent = exponent.to_u32().ok_or(NumericEvaluationError::Limit(
                "integer exponent exceeds the static evaluation budget",
            ))?;
            let estimated = base.bits().saturating_mul(u64::from(exponent));
            if estimated > MAX_INTEGER_RESULT_BITS {
                return Err(NumericEvaluationError::Limit(
                    "integer power exceeds the QCore artifact budget",
                ));
            }
            checked_static(Number::Int(base.pow(exponent)))
        }
        1 => {
            let base = base.into_real()?;
            let exponent = exponent.into_real()?;
            if base == 0.0 && exponent.is_finite() && exponent < 0.0 {
                return Err(NumericEvaluationError::Undefined(
                    "zero cannot be raised to a negative power",
                ));
            }
            if base < 0.0 && exponent.is_finite() && exponent.fract() != 0.0 {
                return Err(NumericEvaluationError::Undefined(
                    "Real power would produce a Complex result",
                ));
            }
            Ok(Number::Real(base.powf(exponent)).into_static())
        }
        _ => {
            let base = base.into_complex()?;
            let exponent = exponent.into_complex()?;
            if base.re == 0.0 && base.im == 0.0 {
                if exponent.im != 0.0 || exponent.re < 0.0 {
                    return Err(NumericEvaluationError::Undefined(
                        "zero cannot be raised to a negative or Complex power",
                    ));
                }
                let result = if exponent.re == 0.0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                return Ok(Number::Complex(result).into_static());
            }
            Ok(Number::Complex(base.powc(exponent)).into_static())
        }
    }
}

#[derive(Clone, Copy)]
enum IntegerBinary {
    And,
    Or,
    Xor,
}

fn integer_binary(
    operands: &[&StaticValue],
    operation: IntegerBinary,
) -> Result<StaticValue, NumericEvaluationError> {
    let [left, right] = integer_operands(operands)?;
    checked_static(Number::Int(match operation {
        IntegerBinary::And => left & right,
        IntegerBinary::Or => left | right,
        IntegerBinary::Xor => left ^ right,
    }))
}

fn shift(
    operands: &[&StaticValue],
    left_shift: bool,
) -> Result<StaticValue, NumericEvaluationError> {
    let [value, distance] = integer_operands(operands)?;
    if distance.is_negative() {
        return Err(NumericEvaluationError::Undefined("negative shift count"));
    }
    let Some(distance) = distance.to_usize() else {
        if left_shift && !value.is_zero() {
            return Err(NumericEvaluationError::Limit(
                "left shift exceeds the QCore artifact budget",
            ));
        }
        let shifted = if value.is_negative() {
            BigInt::from(-1)
        } else {
            BigInt::ZERO
        };
        return Ok(Number::Int(shifted).into_static());
    };
    if left_shift {
        let result_bits = value
            .bits()
            .saturating_add(u64::try_from(distance).unwrap_or(u64::MAX));
        if result_bits > MAX_INTEGER_RESULT_BITS {
            return Err(NumericEvaluationError::Limit(
                "left shift exceeds the QCore artifact budget",
            ));
        }
    }
    checked_static(Number::Int(if left_shift {
        value << distance
    } else {
        value >> distance
    }))
}

fn unary_integer(
    operands: &[&StaticValue],
    operation: impl FnOnce(BigInt) -> BigInt,
) -> Result<StaticValue, NumericEvaluationError> {
    let [value] = operands else {
        return Err(NumericEvaluationError::Undefined(
            "malformed unary integer operation",
        ));
    };
    let value = integer_value(value).ok_or(NumericEvaluationError::Undefined(
        "integer operation received a non-integer",
    ))?;
    checked_static(Number::Int(operation(value)))
}

fn unary_numeric(
    operands: &[&StaticValue],
    negative: bool,
) -> Result<StaticValue, NumericEvaluationError> {
    let [value] = operands else {
        return Err(NumericEvaluationError::Undefined(
            "malformed unary numeric operation",
        ));
    };
    let value = Number::from_static(value).ok_or(NumericEvaluationError::Undefined(
        "numeric operation received a non-numeric value",
    ))?;
    if !negative {
        return Ok(value.into_static());
    }
    checked_static(match value {
        Number::Int(value) => Number::Int(-value),
        Number::Real(value) => Number::Real(-value),
        Number::Complex(value) => Number::Complex(-value),
    })
}

fn cast(
    operation: &wire::OperationProto,
    operands: &[&StaticValue],
) -> Result<StaticValue, NumericEvaluationError> {
    let [value] = operands else {
        return Err(NumericEvaluationError::Undefined("malformed numeric cast"));
    };
    let number = Number::from_static(value).ok_or(NumericEvaluationError::Undefined(
        "numeric cast received a non-numeric value",
    ))?;
    match string_attribute(operation, "target") {
        Some("Bool") => Ok(StaticValue::Scalar(StaticScalar::Bool(!number.is_zero()))),
        Some("Int") => match number {
            Number::Int(value) => checked_static(Number::Int(value)),
            Number::Real(value) if value.is_finite() => {
                let value = BigInt::from_f64(value.trunc()).ok_or(
                    NumericEvaluationError::Undefined("Real cannot be converted to Int"),
                )?;
                checked_static(Number::Int(value))
            }
            Number::Real(_) => Err(NumericEvaluationError::Undefined(
                "non-finite Real cannot be converted to Int",
            )),
            Number::Complex(_) => Err(NumericEvaluationError::Undefined(
                "Complex cannot be converted to Int",
            )),
        },
        Some("Real") => Ok(Number::Real(number.into_real()?).into_static()),
        _ => Err(NumericEvaluationError::Undefined(
            "unknown numeric cast target",
        )),
    }
}

fn numeric_operands(operands: &[&StaticValue]) -> Result<[Number; 2], NumericEvaluationError> {
    let [left, right] = operands else {
        return Err(NumericEvaluationError::Undefined(
            "malformed binary numeric operation",
        ));
    };
    Ok([
        Number::from_static(left).ok_or(NumericEvaluationError::Undefined(
            "numeric operation received a non-numeric value",
        ))?,
        Number::from_static(right).ok_or(NumericEvaluationError::Undefined(
            "numeric operation received a non-numeric value",
        ))?,
    ])
}

fn integer_operands(operands: &[&StaticValue]) -> Result<[BigInt; 2], NumericEvaluationError> {
    let [left, right] = operands else {
        return Err(NumericEvaluationError::Undefined(
            "malformed binary integer operation",
        ));
    };
    Ok([
        integer_value(left).ok_or(NumericEvaluationError::Undefined(
            "integer operation received a non-integer",
        ))?,
        integer_value(right).ok_or(NumericEvaluationError::Undefined(
            "integer operation received a non-integer",
        ))?,
    ])
}

fn integer_value(value: &StaticValue) -> Option<BigInt> {
    match value {
        StaticValue::Scalar(StaticScalar::Int(value)) => Some(BigInt::from(*value)),
        StaticValue::BigInt(value) => Some(BigInt::from_signed_bytes_be(value)),
        _ => None,
    }
}

fn checked_static(value: Number) -> Result<StaticValue, NumericEvaluationError> {
    if let Number::Int(value) = &value
        && value.bits() > MAX_INTEGER_RESULT_BITS
    {
        return Err(NumericEvaluationError::Limit(
            "integer result exceeds the QCore artifact budget",
        ));
    }
    Ok(value.into_static())
}

pub(crate) fn integer_static(value: &BigInt) -> StaticValue {
    value.to_i64().map_or_else(
        || StaticValue::BigInt(value.to_signed_bytes_be().into()),
        |value| StaticValue::Scalar(StaticScalar::Int(value)),
    )
}

fn python_integer_divmod(left: BigInt, right: BigInt) -> (BigInt, BigInt) {
    let mut quotient = &left / &right;
    let mut remainder = left % &right;
    if !remainder.is_zero() && remainder.is_negative() != right.is_negative() {
        quotient -= 1;
        remainder += right;
    }
    (quotient, remainder)
}

fn python_real_divmod(left: f64, right: f64) -> (f64, f64) {
    let mut remainder = left % right;
    let mut quotient = (left - remainder) / right;
    if remainder != 0.0 && remainder.is_sign_negative() != right.is_sign_negative() {
        remainder += right;
        quotient -= 1.0;
    }
    let floor = quotient.floor();
    if quotient - floor > 0.5 {
        quotient = floor + 1.0;
    } else {
        quotient = floor;
    }
    if quotient == 0.0 {
        quotient = 0.0_f64.copysign(left / right);
    }
    if remainder == 0.0 {
        remainder = 0.0_f64.copysign(right);
    }
    (quotient, remainder)
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    let value = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    match value.value.as_ref()? {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unary(opcode: &str, operand: &StaticValue) -> Result<StaticValue, NumericEvaluationError> {
        evaluate(
            &wire::OperationProto {
                opcode: opcode.to_owned(),
                operands: vec![1],
                ..Default::default()
            },
            &[operand],
        )
        .map(|value| value.expect("numeric operation"))
    }

    fn binary(
        opcode: &str,
        left: &StaticValue,
        right: &StaticValue,
    ) -> Result<StaticValue, NumericEvaluationError> {
        evaluate(
            &wire::OperationProto {
                opcode: opcode.to_owned(),
                operands: vec![1, 2],
                ..Default::default()
            },
            &[left, right],
        )
        .map(|value| value.expect("numeric operation"))
    }

    #[test]
    fn integer_divmod_matches_python_sign_rules() {
        for (left, right, quotient, remainder) in [
            (7, 3, 2, 1),
            (-7, 3, -3, 2),
            (7, -3, -3, -2),
            (-7, -3, 2, -1),
        ] {
            assert_eq!(
                python_integer_divmod(BigInt::from(left), BigInt::from(right)),
                (BigInt::from(quotient), BigInt::from(remainder))
            );
        }
    }

    #[test]
    fn real_divmod_matches_python_sign_rules() {
        assert_eq!(python_real_divmod(7.0, 3.0), (2.0, 1.0));
        assert_eq!(python_real_divmod(-7.0, 3.0), (-3.0, 2.0));
        assert_eq!(python_real_divmod(7.0, -3.0), (-3.0, -2.0));
        assert_eq!(python_real_divmod(-7.0, -3.0), (2.0, -1.0));
    }

    #[test]
    fn integer_addition_crosses_i64_without_losing_precision() {
        let value = binary(
            "classical.add",
            &StaticValue::Scalar(StaticScalar::Int(i64::MAX)),
            &StaticValue::Scalar(StaticScalar::Int(1)),
        )
        .expect("arbitrary precision addition");

        assert_eq!(
            value,
            StaticValue::BigInt(
                (BigInt::from(i64::MAX) + BigInt::from(1))
                    .to_signed_bytes_be()
                    .into(),
            )
        );
    }

    #[test]
    fn mixed_complex_multiplication_preserves_binary64_coordinates() {
        let value = binary(
            "classical.mul",
            &StaticValue::ComplexBits(1.0_f64.to_bits(), 2.0_f64.to_bits()),
            &StaticValue::Scalar(StaticScalar::RealBits(3.0_f64.to_bits())),
        )
        .expect("mixed complex multiplication");

        assert_eq!(
            value,
            StaticValue::ComplexBits(3.0_f64.to_bits(), 6.0_f64.to_bits())
        );
    }

    #[test]
    fn square_root_checks_domain_and_preserves_binary64() {
        for value in [0.0_f64, -0.0, 2.0, f64::MAX, f64::MIN_POSITIVE] {
            assert_eq!(
                unary(
                    "classical.sqrt",
                    &StaticValue::Scalar(StaticScalar::RealBits(value.to_bits()))
                ),
                Ok(StaticValue::Scalar(StaticScalar::RealBits(
                    value.sqrt().to_bits()
                )))
            );
        }
        for value in [-1.0_f64, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                unary(
                    "classical.sqrt",
                    &StaticValue::Scalar(StaticScalar::RealBits(value.to_bits()))
                ),
                Err(NumericEvaluationError::Undefined(
                    "sqrt input must be finite and nonnegative"
                ))
            );
        }
        let wide = StaticValue::BigInt((BigInt::from(1) << 100_u32).to_signed_bytes_be().into());
        assert_eq!(
            unary("classical.sqrt", &wide),
            Ok(StaticValue::Scalar(StaticScalar::RealBits(
                2.0_f64.powi(50).to_bits()
            )))
        );
        let excessive =
            StaticValue::BigInt((BigInt::from(1) << 1024_u32).to_signed_bytes_be().into());
        assert!(matches!(
            unary("classical.sqrt", &excessive),
            Err(NumericEvaluationError::Undefined(_))
        ));
    }

    #[test]
    fn structural_math_matches_python_numeric_domains() {
        assert_eq!(
            unary(
                "classical.ceil",
                &StaticValue::Scalar(StaticScalar::RealBits((-1.25_f64).to_bits())),
            ),
            Ok(StaticValue::Scalar(StaticScalar::Int(-1)))
        );
        assert_eq!(
            unary("classical.log2", &StaticValue::Scalar(StaticScalar::Int(8)),),
            Ok(StaticValue::Scalar(StaticScalar::RealBits(
                3.0_f64.to_bits()
            )))
        );
        assert_eq!(
            unary("classical.log2", &StaticValue::Scalar(StaticScalar::Int(0)),),
            Err(NumericEvaluationError::Undefined(
                "log2 input must be finite and strictly positive"
            ))
        );
        let wide_power =
            StaticValue::BigInt((BigInt::from(1) << 4096_u32).to_signed_bytes_be().into());
        assert_eq!(
            unary("classical.log2", &wide_power),
            Ok(StaticValue::Scalar(StaticScalar::RealBits(
                4096.0_f64.to_bits()
            )))
        );
    }

    #[test]
    fn selected_numeric_domain_errors_are_explicit() {
        assert_eq!(
            binary(
                "classical.truediv",
                &StaticValue::Scalar(StaticScalar::Int(1)),
                &StaticValue::BigInt(BigInt::ZERO.to_signed_bytes_be().into()),
            ),
            Err(NumericEvaluationError::Undefined("division by zero"))
        );
        assert_eq!(
            binary(
                "classical.lshift",
                &StaticValue::Scalar(StaticScalar::Int(1)),
                &StaticValue::Scalar(StaticScalar::Int(-1)),
            ),
            Err(NumericEvaluationError::Undefined("negative shift count"))
        );
    }

    #[test]
    fn integer_growth_is_rejected_before_large_allocation() {
        let one = StaticValue::Scalar(StaticScalar::Int(1));
        let two = StaticValue::Scalar(StaticScalar::Int(2));
        let distance = StaticValue::Scalar(StaticScalar::Int(
            i64::try_from(MAX_INTEGER_RESULT_BITS).expect("small test budget"),
        ));

        assert_eq!(
            binary("classical.lshift", &one, &distance),
            Err(NumericEvaluationError::Limit(
                "left shift exceeds the QCore artifact budget"
            ))
        );
        assert_eq!(
            binary("classical.pow", &two, &distance),
            Err(NumericEvaluationError::Limit(
                "integer power exceeds the QCore artifact budget"
            ))
        );
    }

    #[test]
    fn trivial_integer_powers_accept_unbounded_exponents_without_allocation() {
        let exponent = StaticValue::BigInt(
            (BigInt::from(u32::MAX) + BigInt::from(2))
                .to_signed_bytes_be()
                .into(),
        );

        assert_eq!(
            binary(
                "classical.pow",
                &StaticValue::Scalar(StaticScalar::Int(-1)),
                &exponent,
            ),
            Ok(StaticValue::Scalar(StaticScalar::Int(-1)))
        );
        assert_eq!(
            binary(
                "classical.pow",
                &StaticValue::Scalar(StaticScalar::Int(0)),
                &exponent,
            ),
            Ok(StaticValue::Scalar(StaticScalar::Int(0)))
        );
    }

    #[test]
    fn zero_complex_power_matches_python_domains() {
        let zero = StaticValue::ComplexBits(0.0_f64.to_bits(), 0.0_f64.to_bits());
        let zero_exponent = StaticValue::ComplexBits(0.0_f64.to_bits(), 0.0_f64.to_bits());
        let complex_exponent = StaticValue::ComplexBits(1.0_f64.to_bits(), 1.0_f64.to_bits());

        assert_eq!(
            binary("classical.pow", &zero, &zero_exponent),
            Ok(StaticValue::ComplexBits(
                1.0_f64.to_bits(),
                0.0_f64.to_bits()
            ))
        );
        assert_eq!(
            binary("classical.pow", &zero, &complex_exponent),
            Err(NumericEvaluationError::Undefined(
                "zero cannot be raised to a negative or Complex power"
            ))
        );
    }

    #[test]
    fn mixed_integer_real_comparison_does_not_round_the_integer() {
        let integer = StaticValue::BigInt(
            BigInt::from(9_007_199_254_740_993_u64)
                .to_signed_bytes_be()
                .into(),
        );
        let rounded_real =
            StaticValue::Scalar(StaticScalar::RealBits(9_007_199_254_740_993_f64.to_bits()));

        assert_eq!(equal(&integer, &rounded_real), Some(false));
        assert_eq!(order(&integer, &rounded_real), Some(Ordering::Greater));
    }

    #[test]
    fn integer_promotion_rejects_binary64_infinity() {
        let integer =
            StaticValue::BigInt((BigInt::from(1) << 4096_usize).to_signed_bytes_be().into());
        let real = StaticValue::Scalar(StaticScalar::RealBits(1.0_f64.to_bits()));

        assert_eq!(
            binary("classical.add", &integer, &real),
            Err(NumericEvaluationError::Undefined(
                "integer is too large to convert to binary64"
            ))
        );
    }
}
