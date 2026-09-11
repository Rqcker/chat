"""Minimal language-model interface used across CHAT.

Any object with a ``generate(system, user, **params) -> str`` method satisfies
:class:`LLMClient`. This keeps the modules decoupled from a specific SDK and lets
tests inject a fake client.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, system: str, user: str, **params) -> str:
        """Return the model's text response to a system + user prompt."""
        ...
