"""Interactive Facial Behaviour Generation (paper Sec. 4.1, 4.3).

Submodules:
    ifbr/   Interactive Facial Behaviour Refinement (SFBG, RFBG, TCR)

Heavy backbones (talking face, identity generation) are added in their own phases.
``import chat`` and ``import chat.dadg`` stay dependency-light; importing
``chat.ifbg.ifbr`` pulls torch (RFBG). TCR (numpy) and RFBG (torch) are built;
SFBG is planned.
"""
