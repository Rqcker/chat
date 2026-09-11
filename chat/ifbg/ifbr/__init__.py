"""Interactive Facial Behaviour Refinement (paper Sec. 4.3).

Blocks:
    SFBG  Silent Facial Behaviour Generation           [architecture done; train on GPU]
    RFBG  Responsive Facial Behaviour Generation        [architecture done; train on GPU]
    TCR   Temporal Continuity Refinement                [done]
"""

from .rfbg import RFBG, RFBGConfig
from .sfbg import SFBG, SFBGConfig
from .tcr import apply_tcr, blend_boundary, gaussian_alpha

__all__ = [
    "apply_tcr",
    "blend_boundary",
    "gaussian_alpha",
    "RFBG",
    "RFBGConfig",
    "SFBG",
    "SFBGConfig",
]
