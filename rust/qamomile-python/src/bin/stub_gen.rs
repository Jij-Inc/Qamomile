//! Generate the native Python ABI reference from the binding definitions.

fn main() -> pyo3_stub_gen::Result<()> {
    let info = _qamomile_native::stub_info()?;
    let module = &info.modules["qamomile._qamomile_native"];
    let path = info.python_root.join("qamomile/_qamomile_native.pyi");
    let source = module.format_with_config(info.config.use_type_statement);
    std::fs::write(path, format!("{}\n", source.trim_end()))?;
    Ok(())
}
