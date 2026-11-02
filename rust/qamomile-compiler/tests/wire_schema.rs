//! Keep checked-in Rust wire bindings synchronized with canonical Protobuf.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn checked_in_wire_bindings_match_canonical_schemas() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let schema_dir = manifest.join("../../qamomile/proto");
    let schemas = [
        "qcore.proto",
        "hir.proto",
        "resource_analysis.proto",
        "result_postprocessing.proto",
        "circuit_family_selection.proto",
        "execution_snapshot.proto",
        "target_profile.proto",
    ]
    .map(|name| schema_dir.join(name));
    let output = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wire-schema");
    fs::create_dir_all(&output).expect("create schema-check output directory");

    let mut compiler = prost_build::Config::new();
    compiler.out_dir(&output);
    compiler
        .protoc_executable(protoc_bin_vendored::protoc_bin_path().expect("find vendored protoc"))
        .compile_protos(&schemas, &[schema_dir])
        .expect("compile canonical Qamomile schemas");

    let generated_path = output.join("qamomile.proto.rs");
    let format_status = Command::new("rustfmt")
        .args(["--edition", "2024"])
        .arg(&generated_path)
        .status()
        .expect("run rustfmt on generated Rust wire bindings");
    assert!(
        format_status.success(),
        "format generated Rust wire bindings"
    );

    let generated = fs::read(generated_path).expect("read generated Rust wire bindings");
    let checked_in = fs::read(manifest.join("src/artifacts/wire.rs"))
        .expect("read checked-in Rust wire bindings");
    assert_eq!(
        checked_in, generated,
        "canonical Protobuf changed; regenerate rust/qamomile-compiler/src/artifacts/wire.rs"
    );
}
