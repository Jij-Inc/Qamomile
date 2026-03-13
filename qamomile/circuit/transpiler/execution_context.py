"""Execution context for quantum-classical program execution."""

from __future__ import annotations

from typing import Any


class ExecutionContext:
    """Holds global state during program execution."""

    def __init__(self, initial_bindings: dict[str, Any] | None = None):
        self._state: dict[str, Any] = (
            initial_bindings.copy() if initial_bindings else {}
        )

    def get(self, key: str) -> Any:
        return self._state.get(key)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        return {k: self._state[k] for k in keys if k in self._state}

    def update(self, values: dict[str, Any]) -> None:
        self._state.update(values)

    def has(self, key: str) -> bool:
        return key in self._state
