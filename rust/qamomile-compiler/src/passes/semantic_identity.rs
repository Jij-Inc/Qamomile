//! Debug-erased semantic identities for verified `QCore` packages.

use prost::Message;
use sha2::{Digest, Sha256};

use crate::wire;

const SEMANTIC_URI: &str = "qmi://semantic";

/// Return the historical semantic digest of one already verified package.
pub(crate) fn package_digest(package: &wire::QCorePackageProto) -> [u8; 32] {
    let mut semantic = package.clone();
    erase_debug_metadata(&mut semantic);
    semantic.package_digest.clear();
    semantic.package_digest = Sha256::digest(semantic.encode_to_vec()).to_vec();
    Sha256::digest(semantic.encode_to_vec()).into()
}

/// Extend one module identity with its resolved selected callable symbol.
pub(crate) fn selected_digest(module_digest: [u8; 32], symbol: &str) -> [u8; 32] {
    let mut identity = hex_digest(module_digest);
    identity.push(':');
    identity.push_str(symbol);
    Sha256::digest(identity.as_bytes()).into()
}

fn erase_debug_metadata(package: &mut wire::QCorePackageProto) {
    for function in &mut package.functions {
        function.span = Some(semantic_span());
        function.intent.clear();
        for parameter in &mut function.parameters {
            if let Some(value) = parameter.value.as_mut() {
                value.hint.clear();
            }
        }
        let mut blocks = function.body.iter_mut().collect::<Vec<_>>();
        while let Some(block) = blocks.pop() {
            for argument in &mut block.arguments {
                argument.hint.clear();
            }
            for operation in &mut block.operations {
                operation.span = Some(semantic_span());
                operation.intent.clear();
                for result in &mut operation.results {
                    result.hint.clear();
                }
                for region in &mut operation.regions {
                    blocks.extend(region.blocks.iter_mut());
                }
            }
            if let Some(terminator) = block.terminator.as_mut() {
                terminator.span = Some(semantic_span());
            }
        }
    }
}

fn semantic_span() -> wire::SourceSpanProto {
    wire::SourceSpanProto {
        uri: SEMANTIC_URI.to_owned(),
        start_line: 1,
        end_line: 1,
        ..Default::default()
    }
}

fn hex_digest(digest: [u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(result, "{byte:02x}").expect("writing to a String cannot fail");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent_row(name: &str, value: &str) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
            }),
        }
    }

    fn package(with_intent: bool) -> wire::QCorePackageProto {
        let intent = |name: &str| {
            if with_intent {
                vec![intent_row("origin", name), intent_row("doc", "prose")]
            } else {
                Vec::new()
            }
        };
        wire::QCorePackageProto {
            module_name: "identity".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "identity.entry".to_owned(),
                intent: intent("identity.entry"),
                body: Some(wire::BlockProto {
                    operations: vec![wire::OperationProto {
                        opcode: "qcore.constant".to_owned(),
                        intent: intent("identity.entry::constant"),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn intent_rows_change_bytes_but_not_the_semantic_digest() {
        let plain = package(false);
        let annotated = package(true);

        assert_ne!(plain.encode_to_vec(), annotated.encode_to_vec());
        assert_eq!(package_digest(&plain), package_digest(&annotated));
    }
}
