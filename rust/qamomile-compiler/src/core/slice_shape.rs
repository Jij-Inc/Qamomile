//! Bounded dependent slice shapes; the wire spelling is compiler-owned.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use num_bigint::BigInt;

use crate::static_values::normalize_aggregate_slice;
use crate::types::{Dimension, PolynomialDimension};
use crate::wire;

/// A literal or a direct Const Int binder used by one slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SliceBound {
    Omitted,
    Integer(BigInt),
    Binder(String),
}

impl SliceBound {
    pub(crate) fn from_literal(value: &wire::LiteralProto) -> Result<Self, String> {
        use wire::literal_proto::Value;
        match value.value.as_ref() {
            Some(Value::UnitValue(true)) => Ok(Self::Omitted),
            Some(Value::IntValue(value)) => Ok(Self::Integer(BigInt::from(*value))),
            Some(Value::BigintValue(value)) => {
                Ok(Self::Integer(BigInt::from_signed_bytes_be(value)))
            }
            Some(Value::StringValue(name)) if identifier(name) => Ok(Self::Binder(name.clone())),
            _ => {
                Err("slice bounds require integer literals or direct Const Int binders".to_owned())
            }
        }
    }

    fn parse(source: &str) -> Result<Self, String> {
        if source == "~" {
            return Ok(Self::Omitted);
        }
        if identifier(source) {
            return Ok(Self::Binder(source.to_owned()));
        }
        let value = source
            .parse::<BigInt>()
            .map_err(|_| "invalid slice bound")?;
        if value.to_string() != source || source.len() > 128 {
            return Err("slice bound is not a bounded canonical integer".to_owned());
        }
        Ok(Self::Integer(value))
    }

    fn render(&self) -> String {
        match self {
            Self::Omitted => "~".to_owned(),
            Self::Integer(value) => value.to_string(),
            Self::Binder(name) => name.clone(),
        }
    }
}

fn identifier(source: &str) -> bool {
    let mut chars = source.chars();
    chars.next().is_some_and(|c| c == '_' || c.is_alphabetic())
        && chars.all(|c| c == '_' || c.is_alphanumeric())
}

/// A leading extent and the three optional Python slice boundaries.
#[derive(Clone, Debug)]
pub(crate) struct SliceShape {
    pub(crate) extent: String,
    pub(crate) bounds: [SliceBound; 3],
}

impl SliceShape {
    pub(crate) fn render(&self) -> String {
        format!(
            "slice({},{},{},{})",
            self.extent,
            self.bounds[0].render(),
            self.bounds[1].render(),
            self.bounds[2].render()
        )
    }

    pub(crate) fn parse(source: &str) -> Result<Self, String> {
        Self::parse_at(source, 0)
    }

    fn parse_at(source: &str, depth: usize) -> Result<Self, String> {
        if depth >= 8 || source.len() > 4096 {
            return Err("dependent slice shape exceeds its complexity budget".to_owned());
        }
        let row = source
            .strip_prefix("slice(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or("invalid dependent slice shape")?;
        let mut nesting = 0_usize;
        let mut start = 0;
        let mut parts = Vec::new();
        for (index, byte) in row.bytes().enumerate() {
            match byte {
                b'(' => nesting += 1,
                b')' => nesting = nesting.checked_sub(1).ok_or("unbalanced slice shape")?,
                b',' if nesting == 0 => {
                    parts.push(&row[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        parts.push(&row[start..]);
        let [extent, start, stop, stride] = parts.as_slice() else {
            return Err("invalid slice shape arity".to_owned());
        };
        if nesting != 0 {
            return Err("unbalanced slice shape".to_owned());
        }
        if extent.starts_with("slice(") {
            Self::parse_at(extent, depth + 1)?;
        } else if let Ok(value) = extent.parse::<u64>() {
            if value == 0 || value.to_string() != *extent {
                return Err("invalid slice source extent".to_owned());
            }
        } else {
            PolynomialDimension::parse(extent)?;
        }
        let shape = Self {
            extent: (*extent).to_owned(),
            bounds: [
                SliceBound::parse(start)?,
                SliceBound::parse(stop)?,
                SliceBound::parse(stride)?,
            ],
        };
        if shape.render() != source {
            return Err("noncanonical slice shape".to_owned());
        }
        Ok(shape)
    }

    pub(crate) fn symbols(&self) -> BTreeSet<String> {
        let mut symbols = if self.extent.starts_with("slice(") {
            Self::parse(&self.extent)
                .map(|s| s.symbols())
                .unwrap_or_default()
        } else {
            PolynomialDimension::parse(&self.extent)
                .map(|p| p.symbols().map(str::to_owned).collect())
                .unwrap_or_default()
        };
        for bound in &self.bounds {
            if let SliceBound::Binder(name) = bound {
                symbols.insert(name.clone());
            }
        }
        symbols
    }

    pub(crate) fn evaluate(
        &self,
        resolve: &impl Fn(&str) -> Option<i64>,
    ) -> Result<Option<u64>, String> {
        let extent = if self.extent.starts_with("slice(") {
            Self::parse(&self.extent)?.evaluate(resolve)?
        } else if let Ok(value) = self.extent.parse::<u64>() {
            Some(value)
        } else {
            PolynomialDimension::parse(&self.extent)?
                .evaluate_signed(resolve)
                .and_then(|n| u64::try_from(n).ok())
        };
        let Some(extent) = extent else {
            return Ok(None);
        };
        let mut bounds: [Option<BigInt>; 3] = [None, None, None];
        for (target, bound) in bounds.iter_mut().zip(&self.bounds) {
            *target = match bound {
                SliceBound::Omitted => None,
                SliceBound::Integer(value) => Some(value.clone()),
                SliceBound::Binder(name) => {
                    let Some(value) = resolve(name) else {
                        return Ok(None);
                    };
                    Some(BigInt::from(value))
                }
            };
        }
        let [start, stop, stride] = bounds;
        let extent =
            usize::try_from(extent).map_err(|_| "slice source extent is not representable")?;
        let positions =
            normalize_aggregate_slice(extent, start.as_ref(), stop.as_ref(), stride.as_ref())
                .map_err(|e| e.to_string())?;
        if positions.is_empty() {
            return Err("an empty slice has no nonempty fixed-array type".to_owned());
        }
        Ok(Some(positions.len() as u64))
    }

    pub(crate) fn dimension(&self, scope: &Arc<str>) -> Result<Dimension, String> {
        // Validate the generated grammar before admitting it into a type DAG.
        Self::parse(&self.render())?;
        Ok(self.evaluate(&|_| None)?.map_or_else(
            || Dimension::Symbol {
                scope: Arc::clone(scope),
                name: Arc::from(self.render()),
            },
            Dimension::Fixed,
        ))
    }

    pub(crate) fn substitute(
        &self,
        scope: &Arc<str>,
        bindings: &HashMap<Dimension, Dimension>,
    ) -> Dimension {
        let source = self.extent.parse::<u64>().map_or_else(
            |_| Dimension::Symbol {
                scope: Arc::clone(scope),
                name: Arc::from(self.extent.as_str()),
            },
            Dimension::Fixed,
        );
        let source = crate::types::instantiate_dimension(&source, bindings);
        let mut shape = self.clone();
        shape.extent = match source {
            Dimension::Fixed(value) => value.to_string(),
            Dimension::Symbol { name, .. } => name.to_string(),
        };
        for bound in &mut shape.bounds {
            if let SliceBound::Binder(name) = bound {
                let key = Dimension::Symbol {
                    scope: Arc::clone(scope),
                    name: Arc::from(name.as_str()),
                };
                if let Some(value) = bindings.get(&key) {
                    *bound = match value {
                        Dimension::Fixed(value) => SliceBound::Integer(BigInt::from(*value)),
                        Dimension::Symbol { name, .. } => SliceBound::Binder(name.to_string()),
                    };
                }
            }
        }
        shape
            .dimension(scope)
            .unwrap_or_else(|_| Dimension::Symbol {
                scope: Arc::clone(scope),
                name: Arc::from(shape.render()),
            })
    }
}

pub(crate) fn operation_bounds(
    operation: &wire::OperationProto,
) -> Result<[SliceBound; 3], String> {
    if operation.attributes.len() != 3 {
        return Err("invalid slice attribute row".to_owned());
    }
    let mut row = Vec::new();
    for (attribute, expected) in operation.attributes.iter().zip(["start", "step", "stop"]) {
        if attribute.name != expected {
            return Err("invalid slice attribute names".to_owned());
        }
        row.push(SliceBound::from_literal(
            attribute.value.as_ref().ok_or("missing slice bound")?,
        )?);
    }
    let [start, stride, stop]: [SliceBound; 3] =
        row.try_into().map_err(|_| "invalid slice attribute row")?;
    Ok([start, stop, stride])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_shapes_normalize_signed_bounds_and_nested_windows() {
        let shape = SliceShape::parse("slice(slice(5,start,~,1),~,~,-1)").unwrap();
        assert_eq!(shape.symbols(), BTreeSet::from(["start".to_owned()]));
        assert_eq!(
            shape.evaluate(&|name| (name == "start").then_some(-2)),
            Ok(Some(2))
        );
        assert_eq!(shape.evaluate(&|_| None), Ok(None));
        assert_eq!(shape.evaluate(&|_| Some(0)), Ok(Some(5)));
        assert!(
            shape
                .evaluate(&|_| Some(5))
                .unwrap_err()
                .contains("empty slice")
        );
    }

    #[test]
    fn rejects_malformed_or_excessive_shape_encodings() {
        for source in [
            "slice(5,start,~)",
            "slice(5, start,~,1)",
            "slice(05,start,~,1)",
            "slice(5,start,,1)",
        ] {
            assert!(SliceShape::parse(source).is_err(), "{source}");
        }
        let mut source = "n".to_owned();
        for _ in 0..9 {
            source = format!("slice({source},~,~,1)");
        }
        assert!(
            SliceShape::parse(&source)
                .unwrap_err()
                .contains("complexity budget")
        );
        assert!(
            SliceShape::parse("slice(2048,~,~,1)")
                .unwrap()
                .evaluate(&|_| None)
                .unwrap_err()
                .contains("selected items")
        );
    }
}
