"""Keep CUDA-Q out of pytest processes that will not run CUDA-Q tests.

Why this exists: ``cudaq`` imports ``torch`` at import time, and torch,
qiskit-aer, and cudaq each bundle their own copy of LLVM's OpenMP runtime
(``libomp.dylib``). Loading more than one OpenMP runtime into a single
process is undefined behavior; on macOS arm64 it manifests as a
segmentation fault inside AerSimulator's worker threads once the
cudaq/torch libraries are resident. The minimal reproduction is running
``tests/transpiler/backends/test_cudaq.py`` before
``tests/transpiler/test_pauli_evolve_vector_observable.py``: the former's
module-level ``pytest.importorskip("cudaq")`` executes during collection
even when every cudaq test is deselected by the default
``-m "not docs and not quri_parts and not cudaq"`` addopts, and the
latter then crashes inside AerSimulator.

The fix has two halves, enforced by ``tests/test_cudaq_import_isolation.py``:

- ``tests/conftest.py`` skips collecting the modules listed in
  :data:`CUDAQ_MODULE_LEVEL_IMPORTERS` whenever the active marker
  expression cannot select a cudaq-marked test anyway.
- Tests that import cudaq lazily (inside the test body) must carry
  ``pytest.mark.cudaq`` so that default runs never load cudaq
  mid-session either. They still run in the dedicated ``-m cudaq`` CI
  jobs, which execute no qiskit-aer simulations in the same process.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import MutableMapping

NATIVE_THREAD_LIMIT_ENV: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Test modules (rootdir-relative POSIX paths) that import ``cudaq`` at
#: module scope, i.e. during collection. Every entry must also carry
#: ``pytestmark = pytest.mark.cudaq``; tests/test_cudaq_import_isolation.py
#: keeps this table in sync with the actual test sources.
CUDAQ_MODULE_LEVEL_IMPORTERS: frozenset[str] = frozenset(
    {
        "tests/transpiler/backends/test_cudaq.py",
        "tests/transpiler/backends/test_cudaq_frontend.py",
    }
)


@lru_cache(maxsize=None)
def markexpr_can_select_cudaq(markexpr: str) -> bool:
    """Return whether a ``-m`` expression could select a cudaq-marked test.

    Evaluates ``markexpr`` against two hypothetical test items — one whose
    only mark is ``cudaq`` and one carrying every mark — and reports
    whether either would be selected. When both are rejected, no test
    bearing the ``cudaq`` mark can run in this session, so the modules in
    :data:`CUDAQ_MODULE_LEVEL_IMPORTERS` need not be collected at all.

    Args:
        markexpr (str): The raw ``-m`` option value. An empty or blank
            string means no marker filtering, i.e. cudaq tests are
            selectable.

    Returns:
        bool: True when a cudaq-marked test could be selected (collect the
            cudaq modules as usual); False when the expression provably
            deselects all of them.
    """
    expression = markexpr.strip()
    if not expression:
        return True
    try:
        from _pytest.mark.expression import Expression

        compiled = Expression.compile(expression)
        cudaq_only_item = compiled.evaluate(lambda name, **kwargs: name == "cudaq")
        all_marks_item = compiled.evaluate(lambda name, **kwargs: True)
        return cudaq_only_item or all_marks_item
    except Exception:
        # ``Expression`` is private pytest API. If it drifts on a pytest
        # upgrade, recognize only the stock exclusion spelling and
        # otherwise keep collecting as before — never wrongly hide tests.
        return re.search(r"\bnot\s+cudaq\b", expression) is None


def configure_cudaq_thread_limits(
    markexpr: str,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Bound implicit native thread pools when CUDA-Q tests can run.

    Args:
        markexpr (str): Active pytest marker expression.
        environ (MutableMapping[str, str] | None): Environment mapping to
            update. Defaults to ``os.environ``.
    """
    if not markexpr_can_select_cudaq(markexpr):
        return
    target = os.environ if environ is None else environ
    for variable in NATIVE_THREAD_LIMIT_ENV:
        target.setdefault(variable, "1")
