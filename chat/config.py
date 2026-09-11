"""Configuration for the CHAT pipeline.

Defaults live in the dataclasses below; ``configs/default.yaml`` documents the
same fields for users who want to override them. No machine-specific paths or API
keys are stored here. Secrets are read from the environment (see
:class:`LLMConfig.api_key_env`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathsConfig:
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"


@dataclass
class LLMConfig:
    """Settings for the language models (paper: LLM_text and LLM_audio = Gemini)."""

    provider: str = "gemini"
    api_key_env: str = "GEMINI_API_KEY"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"
    timeout: int = 120


@dataclass
class TDGConfig:
    """Textual Dialogue Generation settings (paper Sec. 4.1)."""

    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    num_scripts: int = 1          # N: number of dialogue scripts to generate
    total_turns: int = 8          # turns per conversation (paper 2026-07-01: each dialogue is 5-10 turns; dataset generation may vary within this range)
    segment_min_turns: int = 5    # paper: each segment is 5-10 turns
    segment_max_turns: int = 10
    max_retries: int = 2          # retries when a segment fails to parse/validate


@dataclass
class IARConfig:
    """Interactive Audio Refinement settings (paper Sec. 4.2)."""

    model: str = "gemini-2.5-flash"  # LLM_audio
    temperature: float = 0.3         # low temp for strict structured-JSON refinement
    max_output_tokens: int = 4096
    p_inter: float = 0.3             # P_inter: probability of an interactive word
    iw_duration: float = 0.5         # seconds allotted to an interactive word
    seed: int = 0                    # base seed for interactive-word sampling
    max_retries: int = 2             # retries for LLM_audio refinement
    backchannel_reactivity: float = 0.5  # 0 = neutral back-channels; 1 = full echo of the speaker's emotion


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tdg: TDGConfig = field(default_factory=TDGConfig)
    iar: IARConfig = field(default_factory=IARConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Build a config from defaults, optionally overridden by a YAML file."""
        cfg = cls()
        if not path:
            return cfg
        import yaml  # imported lazily so the package works without PyYAML

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        known = ("paths", "llm", "tdg", "iar")
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        for section in known:
            values = data.get(section)
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"config section {section!r} must be a mapping")
            obj = getattr(cfg, section)
            for key, value in values.items():
                if not hasattr(obj, key):
                    raise ValueError(f"unknown config key {section}.{key}")
                setattr(obj, key, value)
        return cfg
