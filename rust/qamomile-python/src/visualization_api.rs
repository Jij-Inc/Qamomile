//! Read-only projections of the compiler-owned algorithm display.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qamomile_compiler::{
    AlgorithmDisplay, DisplayKind, DisplayNode, DisplayValue, LoopDisplay, VerifiedProgramFamily,
};

use crate::program_owner::program_owner;

/// Immutable, typed port from a transient native display.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct VisualizationValue {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    type_label: String,
    #[pyo3(get)]
    quantum: bool,
    #[pyo3(get)]
    stage: String,
    #[pyo3(get)]
    width: Option<u64>,
}

/// Immutable node with already resolved interfaces and structured children.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct VisualizationNode {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    opcode: String,
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    inputs: Vec<usize>,
    #[pyo3(get)]
    outputs: Vec<usize>,
    #[pyo3(get)]
    children: Vec<VisualizationNode>,
    #[pyo3(get)]
    line: Option<u32>,
}

/// Read-only display returned after verified native projection.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct VisualizationProjection {
    #[pyo3(get)]
    entrypoint: String,
    #[pyo3(get)]
    values: Vec<VisualizationValue>,
    #[pyo3(get)]
    inputs: Vec<usize>,
    #[pyo3(get)]
    nodes: Vec<VisualizationNode>,
}

fn value(value: DisplayValue) -> VisualizationValue {
    VisualizationValue {
        id: value.id,
        label: value.label,
        type_label: value.type_label,
        quantum: value.quantum,
        stage: value.stage,
        width: value.width,
    }
}

fn node(node: DisplayNode) -> VisualizationNode {
    let kind = match node.kind {
        DisplayKind::Quantum => "quantum",
        DisplayKind::Classical => "classical",
        DisplayKind::Structure => "structure",
        DisplayKind::Call => "call",
        DisplayKind::Loop => "loop",
        DisplayKind::Iteration => "iteration",
        DisplayKind::Branch => "branch",
        DisplayKind::Region => "region",
        DisplayKind::Exit => "exit",
    };
    VisualizationNode {
        kind: kind.to_owned(),
        opcode: node.opcode,
        label: node.label,
        inputs: node.inputs,
        outputs: node.outputs,
        children: node.children.into_iter().map(self::node).collect(),
        line: node.line,
    }
}

fn projection(display: AlgorithmDisplay) -> VisualizationProjection {
    VisualizationProjection {
        entrypoint: display.entrypoint,
        values: display.values.into_iter().map(value).collect(),
        inputs: display.inputs,
        nodes: display.nodes.into_iter().map(node).collect(),
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (source, *, family=false, loops="body", inline_depth=1, max_nodes=2048))]
fn project_visualization(
    source: &Bound<'_, PyAny>,
    family: bool,
    loops: &str,
    inline_depth: usize,
    max_nodes: usize,
) -> PyResult<VisualizationProjection> {
    let loops = match loops {
        "folded" => LoopDisplay::Folded,
        "body" => LoopDisplay::Body,
        "expanded" => LoopDisplay::Expanded,
        _ => {
            return Err(PyValueError::new_err(
                "Q7403: loops must be folded, body, or expanded",
            ));
        }
    };
    let result = if family {
        let family = VerifiedProgramFamily::from_bytes(source.extract::<&[u8]>()?)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        family.algorithm_display(loops, inline_depth, max_nodes)
    } else {
        program_owner(source)?
            .verified
            .algorithm_display(loops, inline_depth, max_nodes)
    };
    result
        .map(projection)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<VisualizationValue>()?;
    module.add_class::<VisualizationNode>()?;
    module.add_class::<VisualizationProjection>()?;
    module.add_function(wrap_pyfunction!(project_visualization, module)?)?;
    Ok(())
}
