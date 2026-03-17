"""Import UX tests for the qamomile.cudaq package boundary.

These tests verify that:
- ``import qamomile.cudaq`` alone does not raise even without ``cudaq`` installed.
- Accessing public symbols when ``cudaq`` is missing raises ``ImportError`` with
  actionable install guidance (pip extra, supported platforms, docs URL).
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def _make_find_spec_missing(name: str, *args, **kwargs):
    """Simulate ``cudaq`` being absent from the environment."""
    if name == "cudaq":
        return None
    return importlib.util.find_spec.__wrapped__(name, *args, **kwargs)  # type: ignore[attr-defined]


class TestCudaqImportUX:
    def test_module_import_does_not_raise_without_cudaq(self):
        """``import qamomile.cudaq`` must not raise when cudaq is absent."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            # Re-importing is safe; the module object is already in sys.modules.
            import qamomile.cudaq  # noqa: F401

    def test_symbol_access_raises_import_error_without_cudaq(self):
        """Accessing a public symbol must raise ImportError when cudaq is absent."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            # Remove cached attribute to force __getattr__ to run.
            import qamomile.cudaq as cudaq_mod

            cudaq_mod.__dict__.pop("CudaqTranspiler", None)
            with pytest.raises(ImportError):
                _ = cudaq_mod.CudaqTranspiler

    def test_error_message_contains_pip_extra(self):
        """Error message must mention ``pip install qamomile[cudaq]``."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            import qamomile.cudaq as cudaq_mod

            cudaq_mod.__dict__.pop("CudaqTranspiler", None)
            with pytest.raises(ImportError, match=r"pip install qamomile\[cudaq\]"):
                _ = cudaq_mod.CudaqTranspiler

    def test_error_message_contains_supported_platforms(self):
        """Error message must mention Linux, macOS ARM64, and WSL2."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            import qamomile.cudaq as cudaq_mod

            cudaq_mod.__dict__.pop("CudaqTranspiler", None)
            with pytest.raises(ImportError, match=r"macOS ARM64") as exc_info:
                _ = cudaq_mod.CudaqTranspiler
            msg = str(exc_info.value)
            assert "Linux" in msg
            assert "WSL2" in msg

    def test_error_message_native_windows_unsupported(self):
        """Error message must state that native Windows is not supported."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            import qamomile.cudaq as cudaq_mod

            cudaq_mod.__dict__.pop("CudaqTranspiler", None)
            with pytest.raises(ImportError, match=r"Native Windows is not supported"):
                _ = cudaq_mod.CudaqTranspiler

    def test_error_message_contains_docs_url(self):
        """Error message must contain the official CUDA-Q install docs URL."""
        with patch("importlib.util.find_spec", side_effect=_make_find_spec_missing):
            import qamomile.cudaq as cudaq_mod

            cudaq_mod.__dict__.pop("CudaqTranspiler", None)
            with pytest.raises(ImportError, match=r"nvidia\.github\.io/cuda-quantum"):
                _ = cudaq_mod.CudaqTranspiler

    def test_unknown_attribute_raises_attribute_error(self):
        """Accessing a non-existent attribute must raise AttributeError, not ImportError."""
        import qamomile.cudaq as cudaq_mod

        with pytest.raises(AttributeError):
            _ = cudaq_mod.NonExistentSymbol  # type: ignore[attr-defined]
