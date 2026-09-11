"""Shared helpers for extracting JSON emitted by LLMs."""

from __future__ import annotations

import json

_DECODER = json.JSONDecoder()


def extract_first_json(text: str, opener: str):
    """Return the first valid JSON value beginning with ``opener`` ('[' or '{').

    Scans each candidate start position and uses ``raw_decode`` so trailing prose,
    code fences, and additional brackets after the value do not break parsing.
    """
    if opener not in ("[", "{"):
        raise ValueError("opener must be '[' or '{'")
    start = 0
    while True:
        idx = text.find(opener, start)
        if idx == -1:
            raise ValueError(f"no JSON value starting with {opener!r} in text")
        try:
            value, _ = _DECODER.raw_decode(text[idx:])
            return value
        except json.JSONDecodeError:
            start = idx + 1
