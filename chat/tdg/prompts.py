"""Prompt templates for Textual Dialogue Generation.

The dialogue content is English (matching the paper). Identity descriptors are
short noun phrases such as "a thoughtful gentleman" or "a vibrant lady".
"""

from __future__ import annotations

from ..schema import IdentityDescriptor

IDENTITY_SYSTEM = (
    "You are a casting director. Given a scenario, invent two distinct speakers "
    "for a two-person conversation. Reply with a single JSON object with exactly "
    'two keys, "i" and "j". Each value is a short English noun phrase describing '
    'that speaker, for example {"i": "a thoughtful gentleman", "j": "a vibrant '
    'lady"}. Output only the JSON object.'
)


def identity_user(prompt: str) -> str:
    return f"Scenario: {prompt}\nDescribe speakers i and j."


def dialogue_system(identity_i: IdentityDescriptor, identity_j: IdentityDescriptor) -> str:
    return (
        "You are a dialogue writer. Write a natural, emotionally rich two-person "
        "conversation between speaker A and speaker B that strictly alternates "
        "turns, with each turn responding to the previous one in conversational, "
        "second-person language. Speaker A is " + identity_i.description
        + ". Speaker B is " + identity_j.description + ". The user message states "
        "which speaker begins each block; honour it and then alternate. Reply with "
        'a JSON array; each element is an object {"speaker": "A"|"B", "text": "...", '
        '"emotion": "..."} where emotion is a single word. Output only the JSON array.'
    )


def segment_user(prompt: str, prev_summary: str, num_turns: int, start_index: int) -> str:
    parts = [f"Scenario: {prompt}"]
    if prev_summary:
        parts.append(
            "The conversation so far ends with:\n" + prev_summary +
            "\nContinue naturally from here."
        )
    else:
        parts.append("Begin the conversation.")
    starter = "A" if start_index % 2 == 1 else "B"
    parts.append(
        f"Write the next {num_turns} turns, starting with speaker {starter} and "
        "alternating A, B, A, B."
    )
    return "\n\n".join(parts)
