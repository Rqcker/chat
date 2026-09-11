"""CHAT: Conversational Human Audio-visual Talking Dialogue Generation.

Paper-aligned reimplementation of the CHAT framework (ECCV 2026). The package is
organised to mirror the paper's three modules:

    tdg/   Textual Dialogue Generation               (paper Sec. 4.1)
    dadg/  Dyadic Audio Dialogue Generation + IAR     (paper Sec. 4.1, 4.2)
    ifbg/  Interactive Facial Behaviour Generation + IFBR (paper Sec. 4.1, 4.3)

Nothing here hardcodes machine paths or API keys. Configuration is loaded via
:mod:`chat.config` and secrets come from the environment.
"""

__version__ = "1.0.0"
