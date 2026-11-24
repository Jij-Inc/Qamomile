//! Checked logical shot identities and target context normalization.

use thiserror::Error;

use crate::{CompilerError, VerifiedProgram, wire};

/// Maximum logical indices expanded by one host-bound target invocation.
pub const MAX_INDEXED_SHOTS: u64 = 10_000_000;

/// Aggregate native source, scalar, or result-reconstruction work per invocation.
pub(crate) const MAX_INDEXED_WORK: u128 = 128_000_000;

/// Source bytes processed across repeated host binding, rather than retained memory.
const MAX_INDEXED_SOURCE_BYTES: u128 = 64 * 1024 * 1024 * 1024;

/// Failure to bind or normalize an immutable logical shot context.
#[derive(Debug, Error)]
pub enum CurrentShotError {
    /// The requested index, origin, count or work exceeds the checked domain.
    #[error("Q4480: invalid logical shot context: {0}")]
    Invalid(String),
    /// The bound program failed native semantic verification.
    #[error(transparent)]
    Verification(#[from] CompilerError),
}

/// An invocation-local index whose bounds have been checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogicalShotIndex(u64);

impl LogicalShotIndex {
    /// Check a zero-based index against a positive logical shot count.
    ///
    /// # Errors
    /// Rejects empty domains and indices outside the requested invocation.
    pub fn new(index: u64, shots: u64) -> Result<Self, CurrentShotError> {
        if shots == 0 || index >= shots {
            return Err(CurrentShotError::Invalid(
                "expected 0 <= index < positive shots".into(),
            ));
        }
        Ok(Self(index))
    }

    /// Return the checked logical index.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Certified native origin and the emitter's exact finite integer domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurrentShotPolicy {
    origin: u64,
}

impl CurrentShotPolicy {
    /// Validate a supported native zero-origin or one-origin convention.
    ///
    /// # Errors
    /// Rejects origins outside the explicit supported convention set.
    pub fn new(origin: u64) -> Result<Self, CurrentShotError> {
        if origin > 1 {
            return Err(CurrentShotError::Invalid(
                "native origin must be zero or one".into(),
            ));
        }
        Ok(Self { origin })
    }

    /// Return the native repetition origin.
    #[must_use]
    pub const fn origin(self) -> u64 {
        self.origin
    }

    /// Return the minimum exactly supported target integer.
    #[must_use]
    pub const fn integer_min(self) -> i64 {
        i32::MIN as i64
    }

    /// Return the maximum exactly supported target integer.
    #[must_use]
    pub const fn integer_max(self) -> i64 {
        i32::MAX as i64
    }

    /// Return the maximum invocation count within the target index domain.
    #[must_use]
    pub const fn max_shots(self) -> u64 {
        1_u64 << 31
    }

    /// Validate an invocation before any native execution is submitted.
    ///
    /// # Errors
    /// Rejects zero shots and counts beyond the exact target integer domain.
    pub fn validate(self, shots: u64) -> Result<(), CurrentShotError> {
        if shots == 0 || shots > self.max_shots() {
            return Err(CurrentShotError::Invalid(
                "shot count exceeds the certified native integer domain".into(),
            ));
        }
        Ok(())
    }
}

impl VerifiedProgram {
    /// Check a host-indexed invocation without rewriting or executing source.
    ///
    /// # Errors
    /// Rejects empty invocations and excessive logical-index, source work, or
    /// aggregate processed source bytes.
    pub fn check_indexed_invocation(&self, shots: u64) -> Result<(), CurrentShotError> {
        if shots == 0
            || shots > MAX_INDEXED_SHOTS
            || (self.header().operation_count as u128).saturating_mul(u128::from(shots))
                > MAX_INDEXED_WORK
        {
            return Err(CurrentShotError::Invalid(
                "indexed execution exceeds its native work budget".into(),
            ));
        }
        if (self.canonical_bytes().len() as u128).saturating_mul(u128::from(shots))
            > MAX_INDEXED_SOURCE_BYTES
        {
            return Err(CurrentShotError::Invalid(
                "indexed execution exceeds its native source byte budget".into(),
            ));
        }
        Ok(())
    }

    /// Bind one checked logical index without changing source availability.
    ///
    /// # Errors
    /// Rejects invalid indices or a rewritten program that fails verification.
    pub fn bind_current_shot(&self, index: u64, shots: u64) -> Result<Self, CurrentShotError> {
        let index = LogicalShotIndex::new(index, shots)?;
        self.check_indexed_invocation(shots)?;
        let mut package = self.local.checked.package.clone();
        let value = i64::try_from(index.value()).map_or_else(
            |_| {
                wire::literal_proto::Value::BigintValue(
                    num_bigint::BigInt::from(index.value()).to_signed_bytes_be(),
                )
            },
            wire::literal_proto::Value::IntValue,
        );
        let literal = wire::LiteralProto { value: Some(value) };
        for function in &mut package.functions {
            if let Some(body) = &mut function.body {
                bind_block(body, &literal);
            }
        }
        let bytes = crate::seal_package(package)?;
        Ok(Self::from_bytes(&bytes)?)
    }
}

fn bind_block(block: &mut wire::BlockProto, literal: &wire::LiteralProto) {
    for operation in &mut block.operations {
        if operation.opcode == "qcore.current_shot" {
            "qcore.constant".clone_into(&mut operation.opcode);
            operation.attributes = vec![wire::AttributeProto {
                name: "value".into(),
                value: Some(literal.clone()),
            }];
        }
        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                bind_block(nested, literal);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_and_policy_check_their_observable_domains() {
        assert_eq!(LogicalShotIndex::new(3, 4).unwrap().value(), 3);
        assert!(LogicalShotIndex::new(0, 0).is_err());
        assert!(LogicalShotIndex::new(4, 4).is_err());
        for origin in [0, 1] {
            let policy = CurrentShotPolicy::new(origin).unwrap();
            assert_eq!(policy.origin(), origin);
            assert!(policy.validate(1).is_ok());
            assert!(policy.validate(policy.max_shots()).is_ok());
            assert!(policy.validate(0).is_err());
            assert!(policy.validate(policy.max_shots() + 1).is_err());
        }
        assert!(CurrentShotPolicy::new(2).is_err());
    }
}
