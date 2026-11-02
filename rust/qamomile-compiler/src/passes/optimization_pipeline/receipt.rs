//! Bounded canonical optimization-receipt encoding and verified restoration.

use super::{
    MAX_CONFIGURATION_BYTES, MAX_OPTIMIZATION_PASSES, OptimizationError, OptimizationPassIdentity,
    OptimizationPolicy,
};
use crate::{MAX_QCORE_BYTES, VerifiedProgram};
use sha2::{Digest, Sha256};

const MAX_RECEIPT_BYTES: usize = MAX_QCORE_BYTES + 5 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"QAMOPT01";

pub(super) struct Decoded {
    pub(super) result: VerifiedProgram,
    pub(super) passes: Vec<OptimizationPassIdentity>,
    pub(super) policy: OptimizationPolicy,
}

pub(super) fn encode(
    source: &VerifiedProgram,
    result: &VerifiedProgram,
    passes: &[OptimizationPassIdentity],
    policy: &OptimizationPolicy,
) -> Vec<u8> {
    let mut receipt = MAGIC.to_vec();
    receipt.extend(source.semantic_digest());
    receipt.extend(policy.seed.to_le_bytes());
    receipt.extend(policy.max_work.to_le_bytes());
    put_bytes(&mut receipt, policy.target.as_bytes());
    receipt.extend(
        u32::try_from(passes.len())
            .expect("validated pass count")
            .to_le_bytes(),
    );
    for pass in passes {
        put_bytes(&mut receipt, pass.name.as_bytes());
        put_bytes(&mut receipt, pass.version.as_bytes());
        put_bytes(&mut receipt, &pass.configuration);
    }
    put_bytes(&mut receipt, result.canonical_bytes());
    let digest = Sha256::digest(&receipt);
    receipt.extend_from_slice(&digest);
    receipt
}

pub(super) fn decode(
    source: &VerifiedProgram,
    receipt: &[u8],
) -> Result<Decoded, OptimizationError> {
    if receipt.len() > MAX_RECEIPT_BYTES || receipt.len() < 76 {
        return Err(OptimizationError::invalid(
            "invalid optimization receipt size",
        ));
    }
    let (body, digest) = receipt.split_at(receipt.len() - 32);
    if Sha256::digest(body).as_slice() != digest {
        return Err(OptimizationError::invalid(
            "optimization receipt digest mismatch",
        ));
    }
    let mut reader = Reader(body);
    if reader.take(8)? != MAGIC || reader.take(32)? != source.semantic_digest() {
        return Err(OptimizationError::invalid(
            "optimization receipt source mismatch",
        ));
    }
    let seed = reader.integer()?;
    let max_work = reader.integer()?;
    let target = reader.string(4096)?;
    let policy = OptimizationPolicy::new(seed, max_work, target)?;
    let count = reader.length()?;
    if count > MAX_OPTIMIZATION_PASSES {
        return Err(OptimizationError::invalid("pipeline exceeds 64 passes"));
    }
    let mut passes = Vec::with_capacity(count);
    for _ in 0..count {
        let name = reader.string(128)?;
        let version = reader.string(128)?;
        let configuration = reader.bytes(MAX_CONFIGURATION_BYTES)?.to_vec();
        passes.push(OptimizationPassIdentity::new(name, version, configuration)?);
    }
    let result = VerifiedProgram::from_bytes(reader.bytes(MAX_QCORE_BYTES)?)
        .map_err(|error| OptimizationError::invalid(error.to_string()))?;
    if !reader.0.is_empty() {
        return Err(OptimizationError::invalid(
            "trailing optimization receipt data",
        ));
    }
    Ok(Decoded {
        result,
        passes,
        policy,
    })
}

fn put_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend(
        u32::try_from(bytes.len())
            .expect("native boundary bounds bytes")
            .to_le_bytes(),
    );
    output.extend_from_slice(bytes);
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn integer(&mut self) -> Result<u64, OptimizationError> {
        let bytes = self
            .take(8)?
            .try_into()
            .map_err(|_| OptimizationError::invalid("invalid integer"))?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], OptimizationError> {
        let (value, rest) = self
            .0
            .split_at_checked(count)
            .ok_or_else(|| OptimizationError::invalid("truncated optimization receipt"))?;
        self.0 = rest;
        Ok(value)
    }
    fn length(&mut self) -> Result<usize, OptimizationError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("four-byte slice");
        usize::try_from(u32::from_le_bytes(bytes))
            .map_err(|_| OptimizationError::invalid("length overflow"))
    }
    fn bytes(&mut self, limit: usize) -> Result<&'a [u8], OptimizationError> {
        let count = self.length()?;
        if count > limit {
            return Err(OptimizationError::invalid(
                "optimization field exceeds budget",
            ));
        }
        self.take(count)
    }
    fn string(&mut self, limit: usize) -> Result<String, OptimizationError> {
        String::from_utf8(self.bytes(limit)?.to_vec())
            .map_err(|_| OptimizationError::invalid("invalid UTF-8"))
    }
}
