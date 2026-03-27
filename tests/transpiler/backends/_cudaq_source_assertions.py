"""Helpers for asserting generated CUDA-Q source in backend tests."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from typing import Any

from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.cudaq.emitter import (
    CudaqKernelArtifact,
    CudaqKernelEmitter,
    ExecutionMode,
)
from qamomile.cudaq.transpiler import CudaqEmitPass, CudaqTranspiler

_TRACE_ATTR = "_cudaq_source_trace"


@dataclass(frozen=True)
class EmissionAction:
    """A backend-level emission event captured during code generation."""

    kind: str
    args: tuple[Any, ...]


def _canonical_ast(source: str) -> str:
    """Return a stable AST representation for source comparisons."""
    tree = ast.parse(source)
    return ast.dump(tree, include_attributes=False)


def _angle_expr(angle: Any) -> str:
    """Mirror ``CudaqKernelEmitter._angle_expr`` for source reconstruction."""
    if isinstance(angle, str):
        return angle
    if isinstance(angle, (int, float)):
        return repr(float(angle))
    return str(angle)


class _ExpectedCudaqSourceBuilder:
    """Build expected CUDA-Q source from traced emitter actions."""

    def __init__(
        self,
        *,
        num_qubits: int,
        num_clbits: int,
        boxed_clbits: set[int],
        parametric: bool,
        mode: ExecutionMode,
    ) -> None:
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._boxed_clbits = boxed_clbits
        self._parametric = parametric
        self._mode = mode
        self._indent = 1
        self._lines: list[str] = []

    def build(self, actions: list[EmissionAction]) -> str:
        self._emit(f"q = cudaq.qvector({self._num_qubits})")
        for i in range(self._num_clbits):
            if i in self._boxed_clbits:
                self._emit(f"__b{i} = [False]")
            else:
                self._emit(f"__b{i} = False")

        for action in actions:
            self._emit_action(action)

        if self._mode == ExecutionMode.RUNNABLE:
            clbits = ", ".join(self._clbit_ref(i) for i in range(self._num_clbits))
            self._emit(f"return [{clbits}]")
            if self._parametric:
                signature = "def _qamomile_kernel(thetas: list[float]) -> list[bool]:"
            else:
                signature = "def _qamomile_kernel() -> list[bool]:"
        else:
            if self._parametric:
                signature = "def _qamomile_kernel(thetas: list[float]):"
            else:
                signature = "def _qamomile_kernel():"

        body = "\n".join(self._lines)
        return f"@cudaq.kernel\n{signature}\n{body}\n"

    def _emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line)

    def _clbit_ref(self, idx: int) -> str:
        if idx in self._boxed_clbits:
            return f"__b{idx}[0]"
        return f"__b{idx}"

    def _clbit_store(self, idx: int, expr: str) -> str:
        return f"{self._clbit_ref(idx)} = {expr}"

    def _emit_action(self, action: EmissionAction) -> None:
        kind = action.kind
        args = action.args

        if kind == "h":
            self._emit(f"h(q[{args[0]}])")
            return
        if kind == "x":
            self._emit(f"x(q[{args[0]}])")
            return
        if kind == "y":
            self._emit(f"y(q[{args[0]}])")
            return
        if kind == "z":
            self._emit(f"z(q[{args[0]}])")
            return
        if kind == "s":
            self._emit(f"s(q[{args[0]}])")
            return
        if kind == "sdg":
            self._emit(f"sdg(q[{args[0]}])")
            return
        if kind == "t":
            self._emit(f"t(q[{args[0]}])")
            return
        if kind == "tdg":
            self._emit(f"tdg(q[{args[0]}])")
            return
        if kind == "rx":
            self._emit(f"rx({_angle_expr(args[1])}, q[{args[0]}])")
            return
        if kind == "ry":
            self._emit(f"ry({_angle_expr(args[1])}, q[{args[0]}])")
            return
        if kind == "rz":
            self._emit(f"rz({_angle_expr(args[1])}, q[{args[0]}])")
            return
        if kind == "p":
            self._emit(f"r1({_angle_expr(args[1])}, q[{args[0]}])")
            return
        if kind == "cx":
            self._emit(f"x.ctrl(q[{args[0]}], q[{args[1]}])")
            return
        if kind == "cz":
            self._emit(f"z.ctrl(q[{args[0]}], q[{args[1]}])")
            return
        if kind == "swap":
            self._emit(f"swap(q[{args[0]}], q[{args[1]}])")
            return
        if kind == "cp":
            control, target, angle = args
            angle_expr = _angle_expr(angle)
            if isinstance(angle, (int, float)):
                half = repr(angle / 2.0)
                neg_half = repr(-angle / 2.0)
            else:
                half = f"({angle_expr}) * 0.5"
                neg_half = f"({angle_expr}) * (-0.5)"
            self._emit(f"rz({half}, q[{target}])")
            self._emit(f"x.ctrl(q[{control}], q[{target}])")
            self._emit(f"rz({neg_half}, q[{target}])")
            self._emit(f"x.ctrl(q[{control}], q[{target}])")
            self._emit(f"rz({half}, q[{control}])")
            return
        if kind == "rzz":
            qubit1, qubit2, angle = args
            self._emit(f"x.ctrl(q[{qubit1}], q[{qubit2}])")
            self._emit(f"rz({_angle_expr(angle)}, q[{qubit2}])")
            self._emit(f"x.ctrl(q[{qubit1}], q[{qubit2}])")
            return
        if kind == "ch":
            control, target = args
            self._emit(f"ry({math.pi / 4}, q[{target}])")
            self._emit(f"x.ctrl(q[{control}], q[{target}])")
            self._emit(f"ry({-math.pi / 4}, q[{target}])")
            return
        if kind == "cy":
            control, target = args
            self._emit(f"sdg(q[{target}])")
            self._emit(f"x.ctrl(q[{control}], q[{target}])")
            self._emit(f"s(q[{target}])")
            return
        if kind == "crx":
            control, target, angle = args
            self._emit(f"rx.ctrl({_angle_expr(angle)}, q[{control}], q[{target}])")
            return
        if kind == "cry":
            control, target, angle = args
            self._emit(f"ry.ctrl({_angle_expr(angle)}, q[{control}], q[{target}])")
            return
        if kind == "crz":
            control, target, angle = args
            self._emit(f"rz.ctrl({_angle_expr(angle)}, q[{control}], q[{target}])")
            return
        if kind == "multi_controlled_x":
            controls, target = args
            controls_src = ", ".join(f"q[{index}]" for index in controls)
            if controls_src:
                self._emit(f"x.ctrl({controls_src}, q[{target}])")
            else:
                self._emit(f"x(q[{target}])")
            return
        if kind == "measure":
            qubit, clbit = args
            if self._mode == ExecutionMode.RUNNABLE:
                self._emit(self._clbit_store(clbit, f"mz(q[{qubit}])"))
            return
        if kind == "barrier":
            return
        if kind == "if_start":
            self._emit(f"if {self._clbit_ref(args[0])}:")
            self._indent += 1
            return
        if kind == "else_start":
            self._indent -= 1
            self._emit("else:")
            self._indent += 1
            return
        if kind == "if_end":
            self._indent -= 1
            return
        if kind == "while_start":
            self._emit(f"while {self._clbit_ref(args[0])}:")
            self._indent += 1
            return
        if kind == "while_end":
            self._indent -= 1
            return

        raise AssertionError(f"Unsupported CUDA-Q emission action: {kind}")


def _get_trace(circuit: CudaqKernelArtifact) -> list[EmissionAction]:
    trace = getattr(circuit, _TRACE_ATTR, None)
    if trace is None:
        raise AssertionError("Missing CUDA-Q source trace on circuit artifact.")
    return trace


def assert_traced_source_matches_artifact(
    circuit: CudaqKernelArtifact,
    *,
    mode: ExecutionMode,
    parametric: bool,
    boxed_clbits: set[int],
) -> None:
    """Assert that traced emitter actions reproduce ``circuit.source`` exactly."""
    expected = _ExpectedCudaqSourceBuilder(
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        boxed_clbits=boxed_clbits,
        parametric=parametric,
        mode=mode,
    ).build(_get_trace(circuit))

    expected_ast = _canonical_ast(expected)
    actual_ast = _canonical_ast(circuit.source)
    assert actual_ast == expected_ast, (
        "Generated CUDA-Q source does not match the traced emission contract.\n"
        f"Expected source:\n{expected}\n"
        f"Actual source:\n{circuit.source}"
    )


def assert_inspect_source_matches_artifact(circuit: CudaqKernelArtifact) -> None:
    """Assert that ``inspect.getsource`` resolves to the artifact source."""
    assert circuit.kernel_func is not None, "Expected finalized CUDA-Q kernel."
    kernel_source = inspect.getsource(circuit.kernel_func.kernelFunction)
    assert _canonical_ast(kernel_source) == _canonical_ast(circuit.source), (
        "inspect.getsource() returned source that differs from CudaqKernelArtifact.source.\n"
        f"inspect.getsource():\n{kernel_source}\n"
        f"artifact.source:\n{circuit.source}"
    )


class TracingCudaqKernelEmitter(CudaqKernelEmitter):
    """CUDA-Q emitter that records semantic emission actions for tests."""

    def create_circuit(self, num_qubits: int, num_clbits: int) -> CudaqKernelArtifact:
        circuit = super().create_circuit(num_qubits, num_clbits)
        setattr(circuit, _TRACE_ATTR, [])
        return circuit

    def finalize(
        self, circuit: CudaqKernelArtifact, mode: ExecutionMode
    ) -> CudaqKernelArtifact:
        finalized = super().finalize(circuit, mode)
        assert_traced_source_matches_artifact(
            finalized,
            mode=mode,
            parametric=self._parametric,
            boxed_clbits=set(self._boxed_clbits),
        )
        return finalized

    def _record(self, circuit: CudaqKernelArtifact, kind: str, *args: Any) -> None:
        _get_trace(circuit).append(EmissionAction(kind, args))

    def emit_h(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "h", qubit)
        super().emit_h(circuit, qubit)

    def emit_x(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "x", qubit)
        super().emit_x(circuit, qubit)

    def emit_y(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "y", qubit)
        super().emit_y(circuit, qubit)

    def emit_z(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "z", qubit)
        super().emit_z(circuit, qubit)

    def emit_s(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "s", qubit)
        super().emit_s(circuit, qubit)

    def emit_sdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "sdg", qubit)
        super().emit_sdg(circuit, qubit)

    def emit_t(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "t", qubit)
        super().emit_t(circuit, qubit)

    def emit_tdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        self._record(circuit, "tdg", qubit)
        super().emit_tdg(circuit, qubit)

    def emit_rx(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        self._record(circuit, "rx", qubit, angle)
        super().emit_rx(circuit, qubit, angle)

    def emit_ry(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        self._record(circuit, "ry", qubit, angle)
        super().emit_ry(circuit, qubit, angle)

    def emit_rz(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        self._record(circuit, "rz", qubit, angle)
        super().emit_rz(circuit, qubit, angle)

    def emit_p(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        self._record(circuit, "p", qubit, angle)
        super().emit_p(circuit, qubit, angle)

    def emit_cx(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        self._record(circuit, "cx", control, target)
        super().emit_cx(circuit, control, target)

    def emit_cz(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        self._record(circuit, "cz", control, target)
        super().emit_cz(circuit, control, target)

    def emit_swap(self, circuit: CudaqKernelArtifact, qubit1: int, qubit2: int) -> None:
        self._record(circuit, "swap", qubit1, qubit2)
        super().emit_swap(circuit, qubit1, qubit2)

    def emit_cp(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        self._record(circuit, "cp", control, target, angle)
        super().emit_cp(circuit, control, target, angle)

    def emit_rzz(
        self,
        circuit: CudaqKernelArtifact,
        qubit1: int,
        qubit2: int,
        angle: float | Any,
    ) -> None:
        self._record(circuit, "rzz", qubit1, qubit2, angle)
        super().emit_rzz(circuit, qubit1, qubit2, angle)

    def emit_toffoli(
        self,
        circuit: CudaqKernelArtifact,
        control1: int,
        control2: int,
        target: int,
    ) -> None:
        self._record(circuit, "multi_controlled_x", [control1, control2], target)
        CudaqKernelEmitter.emit_multi_controlled_x(
            self, circuit, [control1, control2], target
        )

    def emit_ch(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        self._record(circuit, "ch", control, target)
        super().emit_ch(circuit, control, target)

    def emit_cy(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        self._record(circuit, "cy", control, target)
        super().emit_cy(circuit, control, target)

    def emit_crx(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        self._record(circuit, "crx", control, target, angle)
        super().emit_crx(circuit, control, target, angle)

    def emit_cry(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        self._record(circuit, "cry", control, target, angle)
        super().emit_cry(circuit, control, target, angle)

    def emit_crz(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        self._record(circuit, "crz", control, target, angle)
        super().emit_crz(circuit, control, target, angle)

    def emit_measure(
        self, circuit: CudaqKernelArtifact, qubit: int, clbit: int
    ) -> None:
        self._record(circuit, "measure", qubit, clbit)
        super().emit_measure(circuit, qubit, clbit)

    def emit_barrier(self, circuit: CudaqKernelArtifact, qubits: list[int]) -> None:
        self._record(circuit, "barrier", tuple(qubits))
        super().emit_barrier(circuit, qubits)

    def emit_multi_controlled_x(
        self,
        circuit: CudaqKernelArtifact,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        self._record(
            circuit,
            "multi_controlled_x",
            list(control_indices),
            target_idx,
        )
        super().emit_multi_controlled_x(circuit, control_indices, target_idx)

    def emit_if_start(
        self, circuit: CudaqKernelArtifact, clbit: int, value: int = 1
    ) -> dict[str, Any]:
        self._record(circuit, "if_start", clbit, value)
        return super().emit_if_start(circuit, clbit, value)

    def emit_else_start(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        self._record(circuit, "else_start")
        super().emit_else_start(circuit, context)

    def emit_if_end(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        self._record(circuit, "if_end")
        super().emit_if_end(circuit, context)

    def emit_while_start(
        self, circuit: CudaqKernelArtifact, clbit: int, value: int = 1
    ) -> dict[str, Any]:
        self._record(circuit, "while_start", clbit, value)
        return super().emit_while_start(circuit, clbit, value)

    def emit_while_end(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        self._record(circuit, "while_end")
        super().emit_while_end(circuit, context)


class ValidatingCudaqEmitPass(CudaqEmitPass):
    """CUDA-Q emit pass wired to the tracing test emitter."""

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> None:
        parametric = bool(parameters)
        emitter = TracingCudaqKernelEmitter(parametric=parametric)
        composite_emitters: list[Any] = []
        StandardEmitPass.__init__(
            self,
            emitter,
            bindings,
            parameters,
            composite_emitters,
        )


class ValidatingCudaqTranspiler(CudaqTranspiler):
    """CUDA-Q transpiler that validates generated source on every transpile."""

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[CudaqKernelArtifact]:
        return ValidatingCudaqEmitPass(bindings, parameters)

    def transpile(self, *args: Any, **kwargs: Any) -> Any:
        """Transpile and verify that inspectable source matches every artifact."""
        executable = super().transpile(*args, **kwargs)
        for compiled in getattr(executable, "compiled_quantum", []):
            circuit = getattr(compiled, "circuit", None)
            if isinstance(circuit, CudaqKernelArtifact):
                assert_inspect_source_matches_artifact(circuit)
        return executable
