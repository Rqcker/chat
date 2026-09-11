"""Textual Dialogue Generation (paper Sec. 4.1).

Generates N diverse multi-turn dyadic dialogue scripts and two identity
descriptors from a single scenario prompt, segment by segment.
"""

from .dialogue import generate_dialogue_scripts, generate_one_script

__all__ = ["generate_dialogue_scripts", "generate_one_script"]
