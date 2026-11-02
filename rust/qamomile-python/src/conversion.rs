//! Shared decimal and digest conversions at the Python boundary.

use num_bigint::{BigInt, BigUint};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Write as _;

pub(super) fn parse_width(value: &str, message: &'static str) -> PyResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| PyValueError::new_err(message))
}

pub(super) fn parse_signed_decimal(value: &str, message: &'static str) -> PyResult<BigInt> {
    BigInt::parse_bytes(value.as_bytes(), 10).ok_or_else(|| PyValueError::new_err(message))
}

pub(super) fn parse_unsigned_decimal(value: &str, message: &'static str) -> PyResult<BigUint> {
    BigUint::parse_bytes(value.as_bytes(), 10).ok_or_else(|| PyValueError::new_err(message))
}

pub(super) fn parse_unsigned_decimals(
    values: &[String],
    message: &'static str,
) -> PyResult<Vec<BigUint>> {
    values
        .iter()
        .map(|value| parse_unsigned_decimal(value, message))
        .collect()
}

pub(super) fn parse_unsigned_decimal_matrix(
    values: Vec<Vec<String>>,
    message: &'static str,
) -> PyResult<Vec<Vec<BigUint>>> {
    values
        .into_iter()
        .map(|row| parse_unsigned_decimals(&row, message))
        .collect()
}

pub(super) fn decimal_rows(values: &[BigUint]) -> Vec<String> {
    values.iter().map(|value| value.to_str_radix(10)).collect()
}

pub(super) fn package_digest_hex(digest: &[u8; 32]) -> String {
    digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    )
}
