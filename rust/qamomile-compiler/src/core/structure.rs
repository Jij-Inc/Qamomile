//! Canonical structural facts that Protobuf re-encoding cannot establish.

use crate::wire;

/// Validate normalized source spans and required function bodies.
pub(crate) fn validate_canonical_structure(package: &wire::QCorePackageProto) -> bool {
    let mut blocks = Vec::new();
    for function in &package.functions {
        if !valid_span(function.span.as_ref()) || !valid_asserted_kind(&function.asserted_kind) {
            return false;
        }
        if function.parameters.iter().any(|parameter| {
            parameter
                .value
                .as_ref()
                .is_some_and(|value| !valid_value(value))
        }) {
            return false;
        }
        let Some(body) = &function.body else {
            return false;
        };
        blocks.push(body);
    }

    while let Some(block) = blocks.pop() {
        let Some(terminator) = &block.terminator else {
            return false;
        };
        if !valid_span(terminator.span.as_ref()) {
            return false;
        }
        if block.arguments.iter().any(|value| !valid_value(value)) {
            return false;
        }
        for operation in &block.operations {
            if !valid_span(operation.span.as_ref()) {
                return false;
            }
            if operation.results.iter().any(|value| !valid_value(value)) {
                return false;
            }
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    true
}

fn valid_asserted_kind(value: &str) -> bool {
    matches!(
        value,
        "" | "classical" | "unitary" | "isometry" | "channel" | "instrument"
    )
}

fn valid_value(value: &wire::ValueProto) -> bool {
    value.has_region == value.region.is_some()
}

fn valid_span(span: Option<&wire::SourceSpanProto>) -> bool {
    span.is_some_and(|value| !value.uri.is_empty() && value.start_line != 0 && value.end_line != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> wire::SourceSpanProto {
        wire::SourceSpanProto {
            uri: "qmi://native-test".to_owned(),
            start_line: 1,
            end_line: 1,
            ..Default::default()
        }
    }

    #[test]
    fn accepts_normalized_function_and_terminator_spans() {
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                body: Some(wire::BlockProto {
                    terminator: Some(wire::TerminatorProto {
                        span: Some(span()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                span: Some(span()),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(validate_canonical_structure(&package));
    }

    #[test]
    fn rejects_a_span_that_would_be_normalized_on_decode() {
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                body: Some(wire::BlockProto {
                    terminator: Some(wire::TerminatorProto {
                        span: Some(wire::SourceSpanProto::default()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                span: Some(span()),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(!validate_canonical_structure(&package));
    }

    #[test]
    fn rejects_hidden_regions_in_every_value_position() {
        let hidden = wire::ValueProto {
            region: Some(wire::RegionRefProto::default()),
            has_region: false,
            ..Default::default()
        };
        let cases = [
            wire::FunctionProto {
                parameters: vec![wire::ParameterProto {
                    value: Some(hidden.clone()),
                    ..Default::default()
                }],
                ..normalized_function()
            },
            wire::FunctionProto {
                body: Some(wire::BlockProto {
                    arguments: vec![hidden.clone()],
                    terminator: Some(normalized_terminator()),
                    ..Default::default()
                }),
                ..normalized_function()
            },
            wire::FunctionProto {
                body: Some(wire::BlockProto {
                    operations: vec![wire::OperationProto {
                        results: vec![hidden],
                        span: Some(span()),
                        ..Default::default()
                    }],
                    terminator: Some(normalized_terminator()),
                    ..Default::default()
                }),
                ..normalized_function()
            },
        ];

        for function in cases {
            let package = wire::QCorePackageProto {
                functions: vec![function],
                ..Default::default()
            };
            assert!(!validate_canonical_structure(&package));
        }
    }

    #[test]
    fn rejects_a_region_flag_without_a_region_message() {
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                parameters: vec![wire::ParameterProto {
                    value: Some(wire::ValueProto {
                        has_region: true,
                        region: None,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..normalized_function()
            }],
            ..Default::default()
        };

        assert!(!validate_canonical_structure(&package));
    }

    #[test]
    fn accepts_every_closed_asserted_callable_kind() {
        for asserted_kind in [
            "",
            "classical",
            "unitary",
            "isometry",
            "channel",
            "instrument",
        ] {
            let package = wire::QCorePackageProto {
                functions: vec![wire::FunctionProto {
                    asserted_kind: asserted_kind.to_owned(),
                    ..normalized_function()
                }],
                ..Default::default()
            };

            assert!(validate_canonical_structure(&package));
        }
    }

    #[test]
    fn rejects_unknown_asserted_callable_kinds() {
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                asserted_kind: "reversible".to_owned(),
                ..normalized_function()
            }],
            ..Default::default()
        };

        assert!(!validate_canonical_structure(&package));
    }

    fn normalized_terminator() -> wire::TerminatorProto {
        wire::TerminatorProto {
            span: Some(span()),
            ..Default::default()
        }
    }

    fn normalized_function() -> wire::FunctionProto {
        wire::FunctionProto {
            body: Some(wire::BlockProto {
                terminator: Some(normalized_terminator()),
                ..Default::default()
            }),
            span: Some(span()),
            ..Default::default()
        }
    }
}
