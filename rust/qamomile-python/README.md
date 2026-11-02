# Qamomile Python bridge

This internal Rust crate exposes `qamomile._qamomile_native` inside the single
`qamomile` Python distribution. Compiler semantics belong to the
`qamomile-compiler` crate. The Cargo crate name is internal and does not define
an independently published Python package.

Repository setup and verification commands are in the root README. Binding
contracts are documented alongside the Rust API modules and the generated
`qamomile/_qamomile_native.pyi` stub. Regenerate it from the repository root:

```bash
cargo run --locked -p qamomile-native --features stub-gen --bin stub_gen
```
