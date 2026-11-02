//! Native typed encoding and decoding for sampled artifact results.

use std::collections::HashMap;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use prost::Message;
use thiserror::Error;

use crate::budget;
use crate::equations::literal_conforms;
use crate::literals::validate_literal_references;
use crate::sample_counts::SampleCountSummary;
use crate::types::{Dimension, ScalarType, SemanticType, Stage, TypeArena, TypeId, WordKind};
use crate::wire;

const MAX_RESULT_ABI_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_VALUE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_BITS: usize = 1_000_000;

/// Failure to encode or decode one typed sampled result.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ResultCodecError {
    /// The persistent result ABI is malformed or internally inconsistent.
    #[error("Q8144: invalid native result ABI: {0}")]
    InvalidAbi(String),
    /// The logical sample has the wrong width or contains a non-bit value.
    #[error("Q8123: ResultABI expects {expected} logical bits, got {actual}")]
    InvalidBits {
        /// Logical width declared by the ABI.
        expected: usize,
        /// Number of supplied logical coordinates.
        actual: usize,
    },
    /// The result type consumes more bits than the logical sample contains.
    #[error("Q8121: logical sample is shorter than ResultABI")]
    ShortSample,
    /// The result type consumes fewer bits than the logical sample contains.
    #[error("Q8124: ResultABI did not consume the logical sample")]
    LongSample,
    /// The declared result type has no sampled-value representation.
    #[error("Q8122: result type is not a sampled result schema")]
    InvalidSchema,
    /// A public typed value cannot be flattened into the declared result type.
    #[error("Q8143: result value does not conform to its ResultABI")]
    InvalidValue,
    /// A big-endian SDK bitstring has the wrong width or characters.
    #[error("Q8103: expected a {expected}-bit native result, got {actual} coordinates")]
    InvalidBigEndianBits {
        /// Native width declared by the ABI.
        expected: usize,
        /// Number of non-space coordinates supplied by the SDK.
        actual: usize,
    },
    /// A q0-first SDK bitstring has the wrong width or characters.
    #[error("Q8104: expected a {expected}-bit native result, got {actual} coordinates")]
    InvalidQ0FirstBits {
        /// Native width declared by the ABI.
        expected: usize,
        /// Number of non-space coordinates supplied by the SDK.
        actual: usize,
    },
}

/// Ordering used by one SDK-native textual result row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBitOrder {
    /// Most-significant/native-high index first, as in Qiskit count keys.
    BigEndian,
    /// Logical qubit zero first, as in Braket count keys.
    Q0First,
}

pub(crate) struct CheckedResultAbi {
    source_order: Box<[u32]>,
    native_width: usize,
    logical_width: usize,
    types: TypeArena,
    result_type: TypeId,
}

/// Decode source-ordered logical bits into one canonical portable literal.
///
/// # Errors
///
/// Returns [`ResultCodecError`] when the ABI is malformed, the bit row has the
/// wrong width, or the declared type is not a pure fixed sampled schema.
pub fn decode_result_value(result_abi: &[u8], bits: &[u8]) -> Result<Vec<u8>, ResultCodecError> {
    let abi = decode_abi(result_abi)?;
    abi.decode_bits(bits)
}

/// Decode and aggregate one canonical logical histogram into typed values.
///
/// # Errors
///
/// Returns [`ResultCodecError`] when the ABI or a logical row is malformed or
/// the declared type is not a pure fixed sampled schema.
pub fn decode_sample_values(
    result_abi: &[u8],
    counts: &SampleCountSummary,
) -> Result<Vec<(Vec<u8>, BigUint)>, ResultCodecError> {
    let abi = decode_abi(result_abi)?;
    let mut positions = HashMap::<Vec<u8>, usize>::new();
    let mut values = Vec::<(Vec<u8>, BigUint)>::new();
    for (bits, count) in counts.entries() {
        let value = abi.decode_bits(bits)?;
        if let Some(index) = positions.get(&value).copied() {
            values[index].1 += count;
        } else {
            positions.insert(value.clone(), values.len());
            values.push((value, count.clone()));
        }
    }
    Ok(values)
}

impl CheckedResultAbi {
    fn decode_bits(&self, bits: &[u8]) -> Result<Vec<u8>, ResultCodecError> {
        Ok(self.decode_literal_bits(bits)?.encode_to_vec())
    }

    pub(crate) fn decode_literal_bits(
        &self,
        bits: &[u8],
    ) -> Result<wire::LiteralProto, ResultCodecError> {
        if bits.len() != self.logical_width || bits.iter().any(|bit| *bit > 1) {
            return Err(ResultCodecError::InvalidBits {
                expected: self.logical_width,
                actual: bits.len(),
            });
        }
        let mut offset = 0_usize;
        let value = decode_type(&self.types, self.result_type, bits, &mut offset)?;
        if offset != bits.len() {
            return Err(ResultCodecError::LongSample);
        }
        Ok(value)
    }
}

/// Encode one canonical portable value into source-ordered logical bits.
///
/// # Errors
///
/// Returns [`ResultCodecError`] when either payload is malformed or the value
/// does not inhabit the fixed sampled schema declared by the ABI.
pub fn encode_result_value(result_abi: &[u8], value: &[u8]) -> Result<Vec<u8>, ResultCodecError> {
    let abi = checked_result_abi(result_abi)?;
    let value = decode_literal(value)?;
    abi.encode_literal(&value)
}

/// Validate one canonical portable value against a complete result ABI.
///
/// Unlike [`encode_result_value`], this accepts every non-quantum value result,
/// not only fixed sampled schemas. It is the final common contract for native
/// Reference values before a language binding constructs its public facade.
///
/// # Errors
///
/// Returns [`ResultCodecError`] when either payload is malformed or the value
/// does not inhabit the result type declared by the ABI.
pub fn validate_result_value(result_abi: &[u8], value: &[u8]) -> Result<(), ResultCodecError> {
    let abi = checked_result_abi(result_abi)?;
    let value = decode_literal(value)?;
    literal_conforms(&abi.types, abi.result_type, &value)
        .then_some(())
        .ok_or(ResultCodecError::InvalidValue)
}

/// Decode one SDK-native textual bit row into source order.
///
/// ASCII spaces used as register separators are ignored. The remaining row
/// must contain exactly the ABI's native width in `0`/`1` characters.
///
/// # Errors
///
/// Returns [`ResultCodecError`] when the ABI or textual row is malformed.
pub fn decode_native_result_bitstring(
    result_abi: &[u8],
    bitstring: &str,
    order: NativeBitOrder,
) -> Result<Vec<u8>, ResultCodecError> {
    let abi = decode_abi(result_abi)?;
    let compact = bitstring
        .bytes()
        .filter(|value| *value != b' ')
        .collect::<Vec<_>>();
    let invalid = compact.len() != abi.native_width
        || compact.iter().any(|value| !matches!(value, b'0' | b'1'));
    if invalid {
        return Err(match order {
            NativeBitOrder::BigEndian => ResultCodecError::InvalidBigEndianBits {
                expected: abi.native_width,
                actual: compact.len(),
            },
            NativeBitOrder::Q0First => ResultCodecError::InvalidQ0FirstBits {
                expected: abi.native_width,
                actual: compact.len(),
            },
        });
    }
    let mut native = compact
        .into_iter()
        .map(|value| value - b'0')
        .collect::<Vec<_>>();
    if order == NativeBitOrder::BigEndian {
        native.reverse();
    }
    abi.source_order
        .iter()
        .map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|index| native.get(index).copied())
                .ok_or_else(|| {
                    ResultCodecError::InvalidAbi(
                        "source order refers outside the native result".to_owned(),
                    )
                })
        })
        .collect()
}

impl CheckedResultAbi {
    pub(crate) fn encode_literal(
        &self,
        value: &wire::LiteralProto,
    ) -> Result<Vec<u8>, ResultCodecError> {
        let mut bits = Vec::with_capacity(self.logical_width);
        encode_type(&self.types, self.result_type, value, &mut bits)?;
        if bits.len() != self.logical_width {
            return Err(ResultCodecError::InvalidValue);
        }
        Ok(bits)
    }
}

pub(crate) fn checked_result_abi(payload: &[u8]) -> Result<CheckedResultAbi, ResultCodecError> {
    decode_abi(payload)
}

fn decode_abi(payload: &[u8]) -> Result<CheckedResultAbi, ResultCodecError> {
    if payload.len() > MAX_RESULT_ABI_BYTES {
        return Err(ResultCodecError::InvalidAbi(
            "result ABI exceeds its byte budget".to_owned(),
        ));
    }
    budget::preflight_result_abi(payload)
        .map_err(|error| ResultCodecError::InvalidAbi(preflight_reason(error)))?;
    let value = wire::ResultAbiProto::decode(payload)
        .map_err(|error| ResultCodecError::InvalidAbi(error.to_string()))?;
    if value.encode_to_vec() != payload {
        return Err(ResultCodecError::InvalidAbi(
            "result ABI is not canonical".to_owned(),
        ));
    }
    let type_spec = value
        .r#type
        .as_ref()
        .ok_or_else(|| ResultCodecError::InvalidAbi("missing result type".to_owned()))?;
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("<artifact-result>");
    let result_type = arena
        .resolve_wire_type(type_spec, &scope)
        .map_err(ResultCodecError::InvalidAbi)?;
    crate::artifact_abi::require_classical(&arena, result_type)
        .map_err(|error| ResultCodecError::InvalidAbi(error.to_string()))?;
    let native_width = usize::try_from(value.native_width)
        .map_err(|_| ResultCodecError::InvalidAbi("native width exceeds usize".to_owned()))?;
    let logical_width = value.source_order.len();
    if logical_width > MAX_RESULT_BITS {
        return Err(ResultCodecError::InvalidAbi(
            "logical result exceeds its bit budget".to_owned(),
        ));
    }
    let sampled_width = arena
        .sampled_width(result_type)
        .map_err(ResultCodecError::InvalidAbi)?;
    let expected_width = sampled_width.and_then(|width| usize::try_from(width).ok());
    let invalid_sample_shape = if logical_width == 0 {
        arena.contains_stage(result_type, Stage::Shot)
            && expected_width.is_some_and(|expected_width| expected_width != 0)
    } else {
        expected_width != Some(logical_width)
    };
    if invalid_sample_shape {
        return Err(ResultCodecError::InvalidAbi(
            "result type disagrees with its logical sample width".to_owned(),
        ));
    }
    if value
        .source_order
        .iter()
        .any(|index| usize::try_from(*index).map_or(true, |index| index >= native_width))
    {
        return Err(ResultCodecError::InvalidAbi(
            "source order refers outside the native result".to_owned(),
        ));
    }
    Ok(CheckedResultAbi {
        source_order: value.source_order.into_boxed_slice(),
        native_width,
        logical_width,
        types: arena,
        result_type,
    })
}

fn decode_literal(payload: &[u8]) -> Result<wire::LiteralProto, ResultCodecError> {
    if payload.len() > MAX_RESULT_VALUE_BYTES {
        return Err(ResultCodecError::InvalidValue);
    }
    budget::preflight_literal(payload).map_err(|_| ResultCodecError::InvalidValue)?;
    let value = wire::LiteralProto::decode(payload).map_err(|_| ResultCodecError::InvalidValue)?;
    if value.encode_to_vec() != payload {
        return Err(ResultCodecError::InvalidValue);
    }
    validate_literal_references(vec![&value]).map_err(|_| ResultCodecError::InvalidValue)?;
    Ok(value)
}

fn decode_type(
    types: &TypeArena,
    type_id: TypeId,
    bits: &[u8],
    offset: &mut usize,
) -> Result<wire::LiteralProto, ResultCodecError> {
    match types.get(type_id) {
        Some(SemanticType::Staged { inner, .. }) => decode_type(types, *inner, bits, offset),
        Some(SemanticType::Scalar(_) | SemanticType::Word { .. } | SemanticType::Fixed { .. }) => {
            decode_leaf(types, type_id, bits, offset)
        }
        Some(SemanticType::Product(elements)) => Ok(tuple_literal(
            elements
                .iter()
                .map(|child| decode_type(types, *child, bits, offset))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(SemanticType::Record { fields, .. }) => {
            let mut entries = fields
                .iter()
                .map(|(name, child)| {
                    Ok(wire::literal_proto::Entry {
                        key: name.clone(),
                        value: Some(decode_type(types, *child, bits, offset)?),
                    })
                })
                .collect::<Result<Vec<_>, ResultCodecError>>()?;
            entries.sort_by(|left, right| left.key.cmp(&right.key));
            Ok(mapping_literal(entries))
        }
        Some(
            SemanticType::Array { .. } | SemanticType::Function { .. } | SemanticType::Quantum(_),
        )
        | None => Err(ResultCodecError::InvalidSchema),
    }
}

fn decode_leaf(
    types: &TypeArena,
    type_id: TypeId,
    bits: &[u8],
    offset: &mut usize,
) -> Result<wire::LiteralProto, ResultCodecError> {
    use wire::literal_proto::Value;

    match types.get(type_id) {
        Some(SemanticType::Scalar(ScalarType::Real)) => {
            let raw = take_bits(bits, offset, 64)?
                .iter()
                .enumerate()
                .fold(0_u64, |value, (index, bit)| {
                    value | (u64::from(*bit) << index)
                });
            let value = f64::from_bits(raw);
            if !value.is_finite() {
                return Err(ResultCodecError::InvalidValue);
            }
            Ok(wire::LiteralProto {
                value: Some(Value::RealValue(value)),
            })
        }
        Some(SemanticType::Scalar(ScalarType::Bit)) => {
            let bit = take_bits(bits, offset, 1)?[0];
            Ok(wire::LiteralProto {
                value: Some(Value::IntValue(i64::from(bit))),
            })
        }
        Some(SemanticType::Scalar(ScalarType::Bool)) => {
            let bit = take_bits(bits, offset, 1)?[0];
            Ok(wire::LiteralProto {
                value: Some(Value::BoolValue(bit != 0)),
            })
        }
        Some(SemanticType::Word { kind, width }) => {
            let width = fixed_dimension(width)?;
            let value = take_bits(bits, offset, width)?;
            match kind {
                WordKind::Bits => Ok(tuple_literal(
                    value
                        .iter()
                        .map(|bit| wire::LiteralProto {
                            value: Some(Value::IntValue(i64::from(*bit))),
                        })
                        .collect(),
                )),
                WordKind::UInt => Ok(integer_literal(&bits_to_bigint(value))),
            }
        }
        Some(SemanticType::Fixed {
            integer_bits,
            fractional_bits,
        }) => {
            let integer_bits = fixed_dimension(integer_bits)?;
            let fractional_bits = fixed_dimension(fractional_bits)?;
            let width = integer_bits
                .checked_add(fractional_bits)
                .ok_or(ResultCodecError::InvalidSchema)?;
            let raw = bits_to_bigint(take_bits(bits, offset, width)?);
            let value = fixed_to_real(&raw, fractional_bits)?;
            Ok(wire::LiteralProto {
                value: Some(Value::RealValue(value)),
            })
        }
        _ => Err(ResultCodecError::InvalidSchema),
    }
}

fn encode_type(
    types: &TypeArena,
    type_id: TypeId,
    value: &wire::LiteralProto,
    bits: &mut Vec<u8>,
) -> Result<(), ResultCodecError> {
    use wire::literal_proto::Value;

    match types.get(type_id) {
        Some(SemanticType::Staged { inner, .. }) => encode_type(types, *inner, value, bits),
        Some(SemanticType::Scalar(_) | SemanticType::Word { .. } | SemanticType::Fixed { .. }) => {
            encode_leaf(types, type_id, value, bits)
        }
        Some(SemanticType::Product(elements)) => {
            let Some(Value::TupleValue(sequence)) = value.value.as_ref() else {
                return Err(ResultCodecError::InvalidValue);
            };
            if sequence.values.len() != elements.len() {
                return Err(ResultCodecError::InvalidValue);
            }
            for (child, child_value) in elements.iter().zip(&sequence.values) {
                encode_type(types, *child, child_value, bits)?;
            }
            Ok(())
        }
        Some(SemanticType::Record { fields, .. }) => {
            let Some(Value::MappingValue(mapping)) = value.value.as_ref() else {
                return Err(ResultCodecError::InvalidValue);
            };
            if mapping.entries.len() != fields.len() {
                return Err(ResultCodecError::InvalidValue);
            }
            let entries = mapping
                .entries
                .iter()
                .map(|entry| (entry.key.as_str(), entry.value.as_ref()))
                .collect::<HashMap<_, _>>();
            for (name, child) in fields {
                let child_value = entries
                    .get(name.as_str())
                    .copied()
                    .flatten()
                    .ok_or(ResultCodecError::InvalidValue)?;
                encode_type(types, *child, child_value, bits)?;
            }
            Ok(())
        }
        Some(
            SemanticType::Array { .. } | SemanticType::Function { .. } | SemanticType::Quantum(_),
        )
        | None => Err(ResultCodecError::InvalidSchema),
    }
}

/// Encode a finite binary64 outcome for the codec and native shot simulator.
pub(crate) fn encode_sampled_real(value: f64) -> Result<[u8; 64], ResultCodecError> {
    if !value.is_finite() {
        return Err(ResultCodecError::InvalidValue);
    }
    Ok(std::array::from_fn(|index| {
        u8::from(value.to_bits() & (1_u64 << index) != 0)
    }))
}

fn encode_leaf(
    types: &TypeArena,
    type_id: TypeId,
    value: &wire::LiteralProto,
    bits: &mut Vec<u8>,
) -> Result<(), ResultCodecError> {
    use wire::literal_proto::Value;

    match types.get(type_id) {
        Some(SemanticType::Scalar(ScalarType::Real)) => {
            let Some(Value::RealValue(value)) = value.value else {
                return Err(ResultCodecError::InvalidValue);
            };
            bits.extend(encode_sampled_real(value)?);
            Ok(())
        }
        Some(SemanticType::Scalar(ScalarType::Bit)) => {
            bits.push(literal_bit(value)?);
            Ok(())
        }
        Some(SemanticType::Scalar(ScalarType::Bool)) => {
            let Some(Value::BoolValue(value)) = value.value else {
                return Err(ResultCodecError::InvalidValue);
            };
            bits.push(u8::from(value));
            Ok(())
        }
        Some(SemanticType::Word { kind, width }) => {
            let width = fixed_dimension(width)?;
            match kind {
                WordKind::Bits => {
                    let Some(Value::TupleValue(sequence)) = value.value.as_ref() else {
                        return Err(ResultCodecError::InvalidValue);
                    };
                    if sequence.values.len() != width {
                        return Err(ResultCodecError::InvalidValue);
                    }
                    bits.extend(
                        sequence
                            .values
                            .iter()
                            .map(literal_bit)
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                WordKind::UInt => {
                    let integer = match value.value.as_ref() {
                        Some(Value::IntValue(value)) => BigInt::from(*value),
                        Some(Value::BigintValue(value)) => BigInt::from_signed_bytes_be(value),
                        _ => return Err(ResultCodecError::InvalidValue),
                    };
                    if integer < BigInt::ZERO
                        || integer.bits() > u64::try_from(width).unwrap_or(u64::MAX)
                    {
                        return Err(ResultCodecError::InvalidValue);
                    }
                    for shift in (0..width).rev() {
                        bits.push(
                            ((&integer >> shift) & BigInt::one())
                                .to_u8()
                                .ok_or(ResultCodecError::InvalidValue)?,
                        );
                    }
                }
            }
            Ok(())
        }
        Some(SemanticType::Fixed {
            integer_bits,
            fractional_bits,
        }) => {
            let integer_bits = fixed_dimension(integer_bits)?;
            let fractional_bits = fixed_dimension(fractional_bits)?;
            let width = integer_bits
                .checked_add(fractional_bits)
                .ok_or(ResultCodecError::InvalidSchema)?;
            let scaled = scaled_fixed_integer(value, fractional_bits)?;
            if scaled < BigInt::ZERO || scaled.bits() > u64::try_from(width).unwrap_or(u64::MAX) {
                return Err(ResultCodecError::InvalidValue);
            }
            for shift in (0..width).rev() {
                bits.push(
                    ((&scaled >> shift) & BigInt::one())
                        .to_u8()
                        .ok_or(ResultCodecError::InvalidValue)?,
                );
            }
            Ok(())
        }
        _ => Err(ResultCodecError::InvalidSchema),
    }
}

fn fixed_dimension(dimension: &Dimension) -> Result<usize, ResultCodecError> {
    let Dimension::Fixed(value) = dimension else {
        return Err(ResultCodecError::InvalidSchema);
    };
    bounded_width(*value)
}

fn bounded_width(value: u64) -> Result<usize, ResultCodecError> {
    let value = usize::try_from(value).map_err(|_| ResultCodecError::InvalidSchema)?;
    (value <= MAX_RESULT_BITS)
        .then_some(value)
        .ok_or(ResultCodecError::InvalidSchema)
}

fn take_bits<'a>(
    bits: &'a [u8],
    offset: &mut usize,
    width: usize,
) -> Result<&'a [u8], ResultCodecError> {
    let end = offset
        .checked_add(width)
        .ok_or(ResultCodecError::ShortSample)?;
    let value = bits
        .get(*offset..end)
        .ok_or(ResultCodecError::ShortSample)?;
    *offset = end;
    Ok(value)
}

fn literal_bit(value: &wire::LiteralProto) -> Result<u8, ResultCodecError> {
    use wire::literal_proto::Value;

    match value.value {
        Some(Value::BoolValue(value)) => Ok(u8::from(value)),
        Some(Value::IntValue(value @ (0 | 1))) => {
            u8::try_from(value).map_err(|_| ResultCodecError::InvalidValue)
        }
        _ => Err(ResultCodecError::InvalidValue),
    }
}

fn scaled_fixed_integer(
    value: &wire::LiteralProto,
    fractional_bits: usize,
) -> Result<BigInt, ResultCodecError> {
    use wire::literal_proto::Value;

    match value.value.as_ref() {
        Some(Value::IntValue(value)) => Ok(BigInt::from(*value) << fractional_bits),
        Some(Value::BigintValue(value)) => {
            Ok(BigInt::from_signed_bytes_be(value) << fractional_bits)
        }
        Some(Value::RealValue(value)) => scale_binary64(*value, fractional_bits),
        _ => Err(ResultCodecError::InvalidValue),
    }
}

fn scale_binary64(value: f64, fractional_bits: usize) -> Result<BigInt, ResultCodecError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ResultCodecError::InvalidValue);
    }
    if value == 0.0 {
        return Ok(BigInt::ZERO);
    }
    let raw = value.to_bits();
    let exponent = i32::try_from((raw >> 52) & 0x7ff).expect("binary64 exponent fits i32");
    let fraction = raw & ((1_u64 << 52) - 1);
    let (mantissa, binary_exponent) = if exponent == 0 {
        (fraction, -1074_i64)
    } else {
        ((1_u64 << 52) | fraction, i64::from(exponent) - 1023 - 52)
    };
    let shift = binary_exponent
        .checked_add(i64::try_from(fractional_bits).map_err(|_| ResultCodecError::InvalidValue)?)
        .ok_or(ResultCodecError::InvalidValue)?;
    let mantissa = BigInt::from(mantissa);
    if shift >= 0 {
        return Ok(mantissa << usize::try_from(shift).map_err(|_| ResultCodecError::InvalidValue)?);
    }
    let right = usize::try_from(-shift).map_err(|_| ResultCodecError::InvalidValue)?;
    let divisor = BigInt::one() << right;
    if &mantissa % &divisor != BigInt::ZERO {
        return Err(ResultCodecError::InvalidValue);
    }
    Ok(mantissa >> right)
}

fn bits_to_bigint(bits: &[u8]) -> BigInt {
    bits.iter()
        .fold(BigInt::ZERO, |value, bit| (value << 1) + BigInt::from(*bit))
}

fn integer_literal(value: &BigInt) -> wire::LiteralProto {
    let value = value.to_i64().map_or_else(
        || wire::literal_proto::Value::BigintValue(value.to_signed_bytes_be()),
        wire::literal_proto::Value::IntValue,
    );
    wire::LiteralProto { value: Some(value) }
}

fn fixed_to_real(value: &BigInt, fractional_bits: usize) -> Result<f64, ResultCodecError> {
    let integer = value >> fractional_bits;
    let integer = integer.to_f64().ok_or(ResultCodecError::InvalidValue)?;
    if fractional_bits == 0 {
        return Ok(integer);
    }
    let mask = (BigInt::one() << fractional_bits) - BigInt::one();
    let fraction = (value & mask)
        .to_f64()
        .ok_or(ResultCodecError::InvalidValue)?;
    let exponent = i32::try_from(fractional_bits).map_err(|_| ResultCodecError::InvalidValue)?;
    let result = integer + fraction * 2_f64.powi(-exponent);
    result
        .is_finite()
        .then_some(result)
        .ok_or(ResultCodecError::InvalidValue)
}

fn tuple_literal(values: Vec<wire::LiteralProto>) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::TupleValue(
            wire::literal_proto::Sequence { values },
        )),
    }
}

fn mapping_literal(entries: Vec<wire::literal_proto::Entry>) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::MappingValue(
            wire::literal_proto::Mapping { entries },
        )),
    }
}

fn preflight_reason(error: budget::PreflightError) -> String {
    match error {
        budget::PreflightError::Budget(reason) | budget::PreflightError::Malformed(reason) => {
            reason
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn staged(stage: &str, inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: stage.to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn dimension(value: u64) -> wire::DimensionProto {
        wire::DimensionProto {
            value: Some(wire::dimension_proto::Value::Fixed(value)),
        }
    }

    fn word(width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![dimension(width)],
            ..Default::default()
        }
    }

    fn uint(width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "UInt".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![dimension(width)],
            ..Default::default()
        }
    }

    fn fixed(integer_bits: u64, fractional_bits: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Fixed.into(),
            name: "UFixed".to_owned(),
            dimensions: vec![dimension(integer_bits), dimension(fractional_bits)],
            ..Default::default()
        }
    }

    fn abi(type_spec: wire::TypeProto, width: u32) -> Vec<u8> {
        wire::ResultAbiProto {
            r#type: Some(type_spec),
            source_order: (0..width).collect(),
            native_width: width,
        }
        .encode_to_vec()
    }

    #[test]
    fn round_trips_nested_record_product_word_and_fixed_results() {
        let result_type = wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: "Answer".to_owned(),
            arguments: vec![
                fixed(1, 3),
                wire::TypeProto {
                    kind: wire::type_proto::Kind::Product.into(),
                    arguments: vec![scalar("Bit"), word(2)],
                    ..Default::default()
                },
            ],
            // Declaration order deliberately differs from canonical map order.
            field_names: vec!["z_fixed".to_owned(), "a_pair".to_owned()],
            ..Default::default()
        };
        let result_abi = abi(result_type, 7);
        let bits = [0, 1, 0, 1, 1, 1, 0];

        let value = decode_result_value(&result_abi, &bits).expect("typed result");

        assert_eq!(
            encode_result_value(&result_abi, &value).expect("source bits"),
            bits
        );
    }

    #[test]
    fn real_sample_round_trip_inherits_shot_through_nested_aggregates() {
        let real = scalar("Real");
        let record = wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: "Score".to_owned(),
            field_names: vec!["value".to_owned()],
            arguments: vec![real.clone()],
            ..Default::default()
        };
        let product = wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: vec![real, record],
            ..Default::default()
        };
        let result_abi = abi(staged("Shot", product), 128);
        let bits = [-3.5_f64, f64::from_bits(1)]
            .into_iter()
            .flat_map(|value| (0..64).map(move |i| u8::from(value.to_bits() & (1_u64 << i) != 0)))
            .collect::<Vec<_>>();
        let value = decode_result_value(&result_abi, &bits).expect("nested real samples");
        assert_eq!(encode_result_value(&result_abi, &value).unwrap(), bits);
    }

    #[test]
    fn real_sample_schema_requires_shot_stage() {
        for ty in [
            scalar("Real"),
            staged("Param", scalar("Real")),
            staged("Const", scalar("Real")),
        ] {
            assert!(decode_result_value(&abi(ty, 64), &[0; 64]).is_err());
        }
    }

    #[test]
    fn real_sample_codec_rejects_nonfinite_values_in_both_directions() {
        let result_abi = abi(staged("Shot", scalar("Real")), 64);
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let literal = wire::LiteralProto {
                value: Some(wire::literal_proto::Value::RealValue(value)),
            }
            .encode_to_vec();
            assert_eq!(
                encode_result_value(&result_abi, &literal),
                Err(ResultCodecError::InvalidValue)
            );
            let bits = (0..64)
                .map(|i| u8::from(value.to_bits() & (1_u64 << i) != 0))
                .collect::<Vec<_>>();
            assert_eq!(
                decode_result_value(&result_abi, &bits),
                Err(ResultCodecError::InvalidValue)
            );
        }
    }

    #[test]
    fn decodes_uint_words_as_unsigned_integers() {
        use wire::literal_proto::Value;

        let result_abi = abi(staged("Shot", uint(3)), 3);
        let bits = [1, 0, 1];
        let value = decode_result_value(&result_abi, &bits).expect("UInt result");
        let decoded = wire::LiteralProto::decode(value.as_slice()).expect("literal");

        assert_eq!(decoded.value, Some(Value::IntValue(5)));
        assert_eq!(
            encode_result_value(&result_abi, &value).expect("source bits"),
            bits
        );
    }

    #[test]
    fn round_trips_uint_words_beyond_i64() {
        use wire::literal_proto::Value;

        let result_abi = abi(staged("Shot", uint(65)), 65);
        let mut bits = vec![0; 65];
        bits[0] = 1;
        bits[64] = 1;
        let value = decode_result_value(&result_abi, &bits).expect("large UInt result");
        let decoded = wire::LiteralProto::decode(value.as_slice()).expect("literal");

        assert!(matches!(decoded.value, Some(Value::BigintValue(_))));
        assert_eq!(
            encode_result_value(&result_abi, &value).expect("source bits"),
            bits
        );
    }

    #[test]
    fn validates_nested_value_results_through_the_complete_abi() {
        use wire::literal_proto::Value;

        let result_type = wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: "Answer".to_owned(),
            arguments: vec![
                scalar("Int"),
                wire::TypeProto {
                    kind: wire::type_proto::Kind::Product.into(),
                    arguments: vec![scalar("Bool"), scalar("Int")],
                    ..Default::default()
                },
            ],
            // Declaration order deliberately differs from canonical map order.
            field_names: vec!["z_total".to_owned(), "a_pair".to_owned()],
            ..Default::default()
        };
        let result_abi = abi(result_type, 0);
        let pair = |first| {
            tuple_literal(vec![
                wire::LiteralProto { value: Some(first) },
                wire::LiteralProto {
                    value: Some(Value::IntValue(3)),
                },
            ])
        };
        let value = |first| {
            mapping_literal(vec![
                wire::literal_proto::Entry {
                    key: "a_pair".to_owned(),
                    value: Some(pair(first)),
                },
                wire::literal_proto::Entry {
                    key: "z_total".to_owned(),
                    value: Some(wire::LiteralProto {
                        value: Some(Value::IntValue(7)),
                    }),
                },
            ])
            .encode_to_vec()
        };

        assert_eq!(
            validate_result_value(&result_abi, &value(Value::BoolValue(true))),
            Ok(())
        );
        assert_eq!(
            validate_result_value(&result_abi, &value(Value::IntValue(1))),
            Err(ResultCodecError::InvalidValue)
        );
    }

    #[test]
    fn distinguishes_value_routes_from_fixed_sample_schemas() {
        let integer = wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(7)),
        }
        .encode_to_vec();
        let boolean = wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BoolValue(true)),
        }
        .encode_to_vec();

        assert_eq!(
            validate_result_value(&abi(staged("Shot", scalar("Int")), 0), &integer),
            Ok(())
        );
        assert_eq!(
            validate_result_value(&abi(scalar("Bool"), 0), &boolean),
            Ok(())
        );
        assert!(matches!(
            checked_result_abi(&abi(staged("Shot", word(2)), 0)),
            Err(ResultCodecError::InvalidAbi(message))
                if message.contains("logical sample width")
        ));
    }

    #[test]
    fn decodes_a_complete_histogram_with_one_checked_abi() {
        let result_abi = abi(word(2), 2);
        let counts = crate::sample_counts::normalize_sample_counts(
            &[
                (vec![1, 0].into_boxed_slice(), BigInt::from(3)),
                (vec![0, 1].into_boxed_slice(), BigInt::from(5)),
            ],
            Some(2),
        )
        .expect("sample summary");

        let values = decode_sample_values(&result_abi, &counts).expect("typed values");

        assert_eq!(values.len(), 2);
        assert_eq!(values[0].1, BigUint::from(5_u8));
        assert_eq!(
            wire::LiteralProto::decode(values[0].0.as_slice())
                .expect("canonical literal")
                .value,
            Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: vec![
                        wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::IntValue(0)),
                        },
                        wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::IntValue(1)),
                        },
                    ],
                }
            ))
        );
    }

    #[test]
    fn rejects_non_bits_wrong_width_and_non_sampled_leaves() {
        let bit_abi = abi(word(2), 2);
        assert_eq!(
            decode_result_value(&bit_abi, &[0]),
            Err(ResultCodecError::InvalidBits {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            decode_result_value(&bit_abi, &[0, 2]),
            Err(ResultCodecError::InvalidBits {
                expected: 2,
                actual: 2,
            })
        );
        assert_eq!(
            decode_result_value(&abi(word(2), 1), &[0]),
            Err(ResultCodecError::InvalidAbi(
                "result type disagrees with its logical sample width".to_owned()
            ))
        );
        assert_eq!(
            decode_result_value(&abi(scalar("Int"), 1), &[0]),
            Err(ResultCodecError::InvalidAbi(
                "result type disagrees with its logical sample width".to_owned()
            ))
        );
    }

    #[test]
    fn preserves_boolean_input_type_safety() {
        let result_abi = abi(scalar("Bool"), 1);
        let integer = wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(1)),
        }
        .encode_to_vec();
        let boolean = wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BoolValue(true)),
        }
        .encode_to_vec();

        assert_eq!(
            encode_result_value(&result_abi, &integer),
            Err(ResultCodecError::InvalidValue)
        );
        assert_eq!(
            decode_result_value(&result_abi, &[1]).expect("Boolean result"),
            boolean
        );
        assert_eq!(
            encode_result_value(&result_abi, &boolean).expect("Boolean result"),
            vec![1]
        );
    }

    #[test]
    fn rejects_source_indices_outside_the_native_result() {
        let malformed = wire::ResultAbiProto {
            r#type: Some(scalar("Bit")),
            source_order: vec![1],
            native_width: 1,
        }
        .encode_to_vec();

        assert!(matches!(
            decode_result_value(&malformed, &[0]),
            Err(ResultCodecError::InvalidAbi(message))
                if message.contains("outside the native result")
        ));
    }

    #[test]
    fn decodes_sdk_text_rows_through_the_result_abi() {
        let result_abi = wire::ResultAbiProto {
            r#type: Some(word(2)),
            source_order: vec![2, 0],
            native_width: 3,
        }
        .encode_to_vec();

        assert_eq!(
            decode_native_result_bitstring(&result_abi, "110", NativeBitOrder::Q0First)
                .expect("q0-first row"),
            vec![0, 1]
        );
        assert_eq!(
            decode_native_result_bitstring(&result_abi, "1 10", NativeBitOrder::BigEndian)
                .expect("big-endian row"),
            vec![1, 0]
        );
    }

    #[test]
    fn rejects_malformed_sdk_text_rows_with_order_specific_diagnostics() {
        let result_abi = abi(word(2), 2);

        assert_eq!(
            decode_native_result_bitstring(&result_abi, "1", NativeBitOrder::BigEndian),
            Err(ResultCodecError::InvalidBigEndianBits {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            decode_native_result_bitstring(&result_abi, "0x", NativeBitOrder::Q0First),
            Err(ResultCodecError::InvalidQ0FirstBits {
                expected: 2,
                actual: 2,
            })
        );
    }
}
