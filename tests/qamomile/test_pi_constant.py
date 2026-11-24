"""Public pi constants share one portable source-language representation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

import qamomile as qm
import qamomile as quantum
from qamomile import pi, pi as circle


@qm.qkernel
def decorated_pi() -> qm.Real:
    """Use the public qualified constant inside a file-backed qkernel."""

    angle = qm.pi / 2
    return angle


@quantum.qkernel
def decorated_alias_pi() -> quantum.Real:
    """Preserve a module alias when capturing one file declaration."""

    return quantum.pi / 2


@qm.qkernel
def decorated_imported_pi() -> qm.Real:
    """Capture the ordinary Python value of an imported public constant."""

    return pi / 2


@qm.qkernel
def decorated_import_alias_pi() -> qm.Real:
    """Capture the ordinary Python value of an aliased public constant."""

    return circle / 2


@qm.qkernel
def decorated_default_pi(angle: qm.Const[qm.Real] = circle) -> qm.Real:
    """Resolve a public constant used only by a parameter default."""

    return angle


def _import_pi_fixture(
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ModuleType]:
    """Import an isolated file so its decorators use real file-backed capture."""

    path = tmp_path / "pi_default_fixture.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pi_default_fixture", path)
    assert spec is not None and spec.loader is not None
    imported = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, imported)
    spec.loader.exec_module(imported)
    return path, imported


def test_pi_is_the_standard_library_float() -> None:
    """Python callers receive math.pi through every public import spelling."""

    assert type(qm.pi) is float
    assert qm.pi == pi == circle == math.pi
    assert "pi" in qm.__all__
    assert not hasattr(qm, "e")
    assert not hasattr(qm, "tau")


@pytest.mark.parametrize(
    ("imports", "expression"),
    [
        ("import qamomile", "qamomile.pi"),
        ("import qamomile as qm", "qm.pi"),
        ("import qamomile as quantum", "quantum.pi"),
        ("from qamomile import pi", "pi"),
        ("from qamomile import pi as circle", "circle"),
    ],
)
def test_pi_imports_work_in_source_and_file_builds(
    imports: str,
    expression: str,
    tmp_path: Path,
) -> None:
    """All explicit import spellings compile as ordinary real literals."""

    source = f"""
import qamomile as qm
{imports}

@qm.qkernel
def angle() -> qm.Real:
    angle = {expression} / 2
    return angle
"""
    path = tmp_path / "pi_kernel.py"
    path.write_text(source, encoding="utf-8")

    from_text = qm.build_source(source, module_name="pi_kernel")
    from_file = qm.build(path, module_name="pi_kernel")

    assert qm.evaluate(from_text.specialize("angle")) == math.pi / 2
    assert qm.evaluate(from_file.specialize("angle")) == math.pi / 2


def test_file_backed_decorators_resolve_pi_imports() -> None:
    """Decorator capture agrees with non-executing source and file builds."""

    for kernel in (
        decorated_pi,
        decorated_alias_pi,
        decorated_imported_pi,
        decorated_import_alias_pi,
    ):
        assert qm.evaluate(kernel.specialize()) == math.pi / 2
    assert qm.evaluate(decorated_default_pi.specialize()) == math.pi


def test_nested_decorators_resolve_module_aliases_and_default_only_imports() -> None:
    """Lexical capture preserves aliases absent from the function body."""

    import qamomile as local_quantum
    from qamomile import pi as local_circle

    @qm.qkernel
    def qualified() -> qm.Real:
        return local_quantum.pi / 2

    @qm.qkernel
    def imported() -> qm.Real:
        return local_circle / 2

    @qm.qkernel
    def defaulted(angle: qm.Const[qm.Real] = local_circle) -> qm.Real:
        return angle

    @qm.qkernel
    def qualified_default(angle: qm.Const[qm.Real] = local_quantum.pi) -> qm.Real:
        return angle

    assert qm.evaluate(qualified.specialize()) == math.pi / 2
    assert qm.evaluate(imported.specialize()) == math.pi / 2
    assert qm.evaluate(defaulted.specialize()) == math.pi
    assert qm.evaluate(qualified_default.specialize()) == math.pi


@pytest.mark.parametrize(
    ("function_name", "parameter_name", "default_expected", "bound_expected"),
    [
        ("parameter", "pi", math.pi, 0.5),
        ("aliased_parameter", "local_circle", math.pi, 0.5),
        ("local", "angle", 2.0 + math.pi, 2.5),
    ],
)
def test_nested_default_capture_preserves_same_named_body_bindings(
    function_name: str,
    parameter_name: str,
    default_expected: float,
    bound_expected: float,
) -> None:
    """Signature names refer outward while the body retains its own bindings."""

    from qamomile import pi as local_circle

    @qm.qkernel
    def parameter(pi: qm.Const[qm.Real] = pi) -> qm.Real:
        return pi

    @qm.qkernel
    def aliased_parameter(
        local_circle: qm.Const[qm.Real] = local_circle,
    ) -> qm.Real:
        return local_circle

    @qm.qkernel
    def local(angle: qm.Const[qm.Real] = local_circle) -> qm.Real:
        local_circle = 2.0
        return local_circle + angle

    kernel = {
        "parameter": parameter,
        "aliased_parameter": aliased_parameter,
        "local": local,
    }[function_name]
    assert qm.evaluate(kernel.specialize()) == default_expected
    assert qm.evaluate(kernel.specialize(**{parameter_name: 0.5})) == bound_expected


@pytest.mark.parametrize("expression", ["qm.pi", "circle", "PI", "ALIAS"])
def test_pi_is_portable_in_constants_and_defaults(expression: str) -> None:
    """Declarations reuse the same literal path as qkernel expressions."""

    source = f"""
from typing import NamedTuple

import qamomile as qm
from qamomile import pi as circle

PI = qm.pi
ALIAS: qm.Const[qm.Real] = PI

class Angles(NamedTuple):
    angle: qm.Real = {expression}

@qm.qkernel
def const_default(angle: qm.Const[qm.Real] = {expression}) -> qm.Real:
    return angle

@qm.qkernel
def param_default(angle: qm.Param[qm.Real] = {expression}) -> qm.Real:
    return angle

@qm.qkernel
def record_default() -> qm.Real:
    return Angles().angle

@qm.qkernel
def constant() -> qm.Real:
    return ALIAS
"""
    module = qm.build_source(source)

    for name in ("const_default", "param_default", "record_default", "constant"):
        assert qm.evaluate(module.specialize(name)) == math.pi
    assert (
        module.specialize("const_default").to_bytes()
        == module.specialize("const_default", angle=math.pi).to_bytes()
    )
    assert qm.evaluate(module.specialize("param_default"), angle=0.5) == 0.5


def test_local_names_and_record_fields_shadow_pi_imports() -> None:
    """The public constant cannot replace a local value or record projection."""

    source = """
from typing import NamedTuple

import qamomile as qm
import qamomile as quantum
from qamomile import pi

class Angles(NamedTuple):
    pi: qm.Real

@qm.qkernel
def parameter(pi: qm.Param[qm.Real]) -> qm.Real:
    return pi

@qm.qkernel
def local() -> qm.Real:
    pi = 2.0
    return pi

@qm.qkernel
def field(quantum: Angles) -> qm.Real:
    return quantum.pi

@qm.qkernel
def local_record() -> qm.Real:
    quantum = Angles(4.0)
    return quantum.pi
"""
    module = qm.build_source(source)

    assert qm.evaluate(module.specialize("parameter"), pi=1.0) == 1.0
    assert qm.evaluate(module.specialize("local")) == 2.0
    assert qm.evaluate(module.specialize("field"), quantum={"pi": 3.0}) == 3.0
    assert qm.evaluate(module.specialize("local_record")) == 4.0


def test_module_constant_rebinding_shadows_an_imported_pi() -> None:
    """A later module constant remains the value selected by bare-name lookup."""

    module = qm.build_source("""
import qamomile as qm
from qamomile import pi

pi = 2.0

@qm.qkernel
def value(angle: qm.Const[qm.Real] = pi) -> qm.Real:
    return pi + angle
""")

    assert qm.evaluate(module.specialize("value")) == 4.0


def test_imported_pi_replaces_a_previous_module_constant() -> None:
    """A later explicit import selects pi instead of retaining a stale value."""

    module = qm.build_source("""
import qamomile as qm

pi = 2.0
from qamomile import pi

@qm.qkernel
def value(angle: qm.Const[qm.Real] = pi) -> qm.Real:
    return pi + angle
""")

    assert qm.evaluate(module.specialize("value")) == 2 * math.pi


@pytest.mark.parametrize(
    ("declarations", "expected"),
    [
        (
            """
@qm.qkernel
def value(angle: qm.Const[qm.Real] = pi) -> qm.Real:
    return angle

pi = 2.0
""",
            math.pi,
        ),
        (
            """
class Angles(NamedTuple):
    angle: qm.Real = pi

pi = 2.0

@qm.qkernel
def value() -> qm.Real:
    return Angles().angle
""",
            math.pi,
        ),
        (
            """
pi: qm.Real

@qm.qkernel
def value() -> qm.Real:
    return pi
""",
            math.pi,
        ),
        (
            """
import qamomile
import qamomile.types

@qm.qkernel
def value() -> tuple[qm.Real, qm.Real]:
    return qamomile.pi, qamomile.sqrt(4.0)
""",
            (math.pi, 2.0),
        ),
        (
            """
class Angles(NamedTuple):
    pi: qm.Real = 1.0
    angle: qm.Real = pi

@qm.qkernel
def value() -> qm.Real:
    return Angles().angle
""",
            1.0,
        ),
        (
            """
import qamomile as quantum

@qm.qkernel
def value(angle: qm.Const[qm.Real] = quantum.pi) -> qm.Real:
    return angle

quantum = 2.0
""",
            math.pi,
        ),
    ],
    ids=[
        "function-default-before-rebinding",
        "record-default-before-rebinding",
        "annotation-only",
        "dotted-import-preserves-root",
        "record-default-uses-earlier-field",
        "qualified-default-before-rebinding",
    ],
)
def test_pi_declaration_binding_matches_file_decorator_capture(
    declarations: str,
    expected: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults bind at declaration time, and annotations do not rebind names."""

    source = f"""
from typing import NamedTuple

import qamomile as qm
from qamomile import pi

{declarations}
"""
    path, imported = _import_pi_fixture(source, tmp_path, monkeypatch)

    from_source = qm.build_source(source).specialize("value")
    from_file = qm.build(path).specialize("value")
    from_decorator = imported.value.specialize()

    assert qm.evaluate(from_source) == expected
    assert qm.evaluate(from_file) == expected
    assert qm.evaluate(from_decorator) == expected


@pytest.mark.parametrize(
    ("expression", "type_source", "expected", "module_constant"),
    [
        ("-qm.pi", "qm.Real", -math.pi, False),
        ("-circle", "qm.Real", -math.pi, False),
        ("(qm.pi,)", "tuple[qm.Real]", (math.pi,), False),
        ("(circle,)", "tuple[qm.Real]", (math.pi,), False),
        ("-qm.pi", "qm.Real", -math.pi, True),
        ("-circle", "qm.Real", -math.pi, True),
        ("+qm.pi", "qm.Real", math.pi, True),
        ("+circle", "qm.Real", math.pi, True),
    ],
)
def test_pi_preserves_portable_unary_and_tuple_literal_syntax(
    expression: str,
    type_source: str,
    expected: object,
    module_constant: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Literal containers and signs agree across source and decorator capture."""

    constant = f"VALUE = {expression}" if module_constant else ""
    default = "VALUE" if module_constant else expression
    source = f"""
import qamomile as qm
from qamomile import pi as circle

{constant}

@qm.qkernel
def value(angle: qm.Const[{type_source}] = {default}) -> {type_source}:
    return angle
"""
    path, imported = _import_pi_fixture(source, tmp_path, monkeypatch)

    assert qm.evaluate(qm.build_source(source).specialize("value")) == expected
    assert qm.evaluate(qm.build(path).specialize("value")) == expected
    assert qm.evaluate(imported.value.specialize()) == expected


def test_pi_rejects_aggregate_field_alias_expansion() -> None:
    """Repeated aggregate aliases fail before materializing an exponential value."""

    fields = ["    field0: tuple[qm.Real] = (qm.pi,)"]
    fields.extend(
        f"    field{index}: qm.Real = (field{index - 1}, field{index - 1})"
        for index in range(1, 30)
    )
    source = "import qamomile as qm\nfrom typing import NamedTuple\n\n"
    source += "class Angles(NamedTuple):\n" + "\n".join(fields)

    with pytest.raises(qm.FrontendError, match="Q5099"):
        qm.build_source(source)


def test_pi_ignores_ordinary_class_aggregate_initializers() -> None:
    """Unreferenced Python classes cannot trigger source constant expansion."""

    fields = ["    field0: tuple[qm.Real] = (qm.pi,)"]
    fields.extend(
        f"    field{index}: qm.Real = (field{index - 1}, field{index - 1})"
        for index in range(1, 30)
    )
    source = "import qamomile as qm\n\nclass Ordinary:\n" + "\n".join(fields)
    source += "\n\n@qm.qkernel\ndef value() -> qm.Real:\n    return qm.pi\n"

    assert qm.evaluate(qm.build_source(source).specialize("value")) == math.pi


@pytest.mark.parametrize("expression", ["pi", "quantum.pi"])
def test_later_local_binding_does_not_turn_an_unbound_read_into_pi(
    expression: str,
) -> None:
    """Function-local scope applies even before a local assignment executes."""

    with pytest.raises(qm.FrontendError, match="Q5020"):
        qm.build_source(f"""
import qamomile as qm
import qamomile as quantum
from qamomile import pi

@qm.qkernel
def value() -> qm.Real:
    angle = {expression}
    pi = 2.0
    quantum = 3.0
    return angle
""")


def test_rebound_module_alias_does_not_resolve_to_pi() -> None:
    """A scalar that hides a module alias cannot provide a pi attribute."""

    with pytest.raises(qm.FrontendError, match="Q5025"):
        qm.build_source("""
import qamomile as qm
import qamomile as quantum

quantum = 2.0

@qm.qkernel
def value() -> qm.Real:
    return quantum.pi
""")


@pytest.mark.parametrize(
    ("imports", "expression"),
    [
        ("import math", "math.pi"),
        ("import numpy as np", "np.pi"),
        ("import unrelated as other", "other.pi"),
        ("", "qm.e"),
        ("", "qm.tau"),
    ],
)
def test_pi_does_not_enable_arbitrary_module_attributes(
    imports: str,
    expression: str,
) -> None:
    """Only the explicitly supported Qamomile constant gains source meaning."""

    with pytest.raises(qm.FrontendError, match="Q5020"):
        qm.build_source(f"""
import qamomile as qm
{imports}

@qm.qkernel
def value() -> qm.Real:
    return {expression}
""")


@pytest.mark.parametrize("expression", ["qm.pi / 2", "circle / 2"])
def test_pi_does_not_enable_arithmetic_parameter_defaults(
    expression: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default expressions retain the existing portable-literal restriction."""

    source = f"""
import qamomile as qm
from qamomile import pi as circle

@qm.qkernel
def value(angle: qm.Const[qm.Real] = {expression}) -> qm.Real:
    return angle
"""
    path, imported = _import_pi_fixture(source, tmp_path, monkeypatch)

    with pytest.raises(qm.FrontendError, match="Q5007"):
        qm.build_source(source)
    with pytest.raises(qm.FrontendError, match="Q5007"):
        qm.build(path)
    with pytest.raises(qm.FrontendError, match="Q5007"):
        imported.value.specialize()


def test_deep_arithmetic_default_is_rejected_without_recursive_substitution() -> None:
    """Unsupported arithmetic defaults fail before walking their deep expression."""

    expression = "qm.pi + " * 1200 + "0.0"
    source = f"""
import qamomile as qm

@qm.qkernel
def value(angle: qm.Const[qm.Real] = {expression}) -> qm.Real:
    return angle
"""

    with pytest.raises(qm.FrontendError, match="Q5007"):
        qm.build_source(source)


def test_pi_survives_module_and_program_persistence() -> None:
    """Saved programs retain the real value without resolving Python imports."""

    module = qm.build_source("""
import qamomile as qm

PI = qm.pi

@qm.qkernel
def value(angle: qm.Const[qm.Real] = PI) -> qm.Real:
    return angle / 2
""")
    restored_module = qm.Module.from_bytes(module.to_bytes())
    program = restored_module.specialize("value")
    restored_program = qm.Program.from_bytes(program.to_bytes())

    assert restored_module.to_bytes() == module.to_bytes()
    assert restored_program.to_bytes() == program.to_bytes()
    assert qm.evaluate(restored_program) == math.pi / 2


def test_pi_rotation_executes_on_the_reference_runtime() -> None:
    """A pi rotation changes a measured zero qubit to one deterministically."""

    program = qm.build_source("""
import qamomile as qm

@qm.qkernel
def rotate() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = qm.rx(state, qm.pi)
    return qm.measure(state)
""").specialize("rotate")
    restored = qm.Program.from_bytes(program.to_bytes())

    result = qm.run(restored, on=qm.Reference(), shots=4, seed=7).result

    assert result.counts == {(1,): 4}
