# Changelog

Versions follow [semantic versioning](https://semver.org). Measurements for a
release live in that release's notes, not here, because they depend on the
protocol of the version that produced them.

## v1.0.0 — 2026-09-11

First public release.

**Pipeline.** Six stages from one text prompt to a paired dyadic clip:
dialogue and identity descriptors (TDG), dyadic speech on a shared timeline with
interactive audio refinement (DADG/IAR), identity frames, talking-head rendering,
interactive facial behaviour refinement (RFBG, SFBG, TCR), and composition.
`scripts/x1_gen_core.sh` chains stages 1 to 5.

**Weights.** The two IFBR refinement blocks, `rfbg_qknorm_ft_030500.pt` and
`sfbg_pre_020000.pt`, each carrying its own config so the architecture is read
from the file. Released for non-commercial research use under
`checkpoints/LICENSE`.

**Evaluation.** FID under two reference protocols, FVD, LSE-C/LSE-D, CSIM,
emotion accuracy and MCD, each with the settings it depends on documented in
`DOCS.md`. Reproduction figures are in the release notes.

**Tests.** 119 unit tests covering the schema, the dialogue planner, the IAR
encoders and timeline, the RFBG and SFBG denoisers, TCR, the metric implementations
and the silence detector. No GPU and no network required.

**Documentation.** `DOCS.md` for implementation choices, how each module relates
to the paper's description, and the evaluation protocol. `README.md` for install
and first run. Responsible-use terms in `README.md`.
