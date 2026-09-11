"""X1 step 1 (chat env): scene prompt -> dialogue script + interactive audio plan.

Runs TDG (LLM dialogue + identities) and the deterministic+LLM parts of IAR
(organise units, sample interactive words, refine text/emotion/sound environment).
Timestamps are NOT assigned here — they need real unit durations, which the TTS
step measures and back-fills.

Usage (needs the provider key in the env, e.g. DEEPSEEK_API_KEY):
  python scripts/x1_text.py --prompt "..." --out x1_out \
      [--provider openai_compat --endpoint https://api.deepseek.com \
       --api-key-env DEEPSEEK_API_KEY --model deepseek-chat --turns 4]
"""

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.config import Config  # noqa: E402
from chat.dadg.iar.interactive_words import sample_interactive_words  # noqa: E402
from chat.dadg.iar.refine import refine_audio_plan, refine_dialogue  # noqa: E402
from chat.dadg.organise import build_audio_units  # noqa: E402
from chat.llm import build_llm  # noqa: E402
from chat.schema import InteractiveAudioPlan  # noqa: E402
from chat.tdg.dialogue import generate_dialogue_scripts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="openai_compat")
    ap.add_argument("--endpoint", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--turns", type=int, default=8)  # paper: each conversation 5-10 turns
    args = ap.parse_args()

    cfg = Config()
    cfg.llm.provider = args.provider
    cfg.llm.endpoint = args.endpoint
    cfg.llm.api_key_env = args.api_key_env
    cfg.tdg.model = args.model
    cfg.iar.model = args.model
    cfg.tdg.total_turns = args.turns
    cfg.tdg.segment_min_turns = min(cfg.tdg.segment_min_turns, args.turns)

    os.makedirs(args.out, exist_ok=True)
    llm = build_llm(cfg, role="tdg")

    print("== TDG: generating dialogue ==")
    scripts = generate_dialogue_scripts(args.prompt, llm, cfg)
    script = scripts[0]
    script.save(os.path.join(args.out, "script.json"))
    print(f"turns: {script.num_turns}")
    for t in script.turns:
        print(f"  [{t.index}] S^{t.speaker}: {t.text[:70]}")
    print(f"identity_i: {script.identity_i}")
    print(f"identity_j: {script.identity_j}")

    print("== IAR: organise + interactive words + LLM refine ==")
    iws = sample_interactive_words(script.num_turns, cfg.iar.p_inter, seed=cfg.iar.seed + script.index)
    units_i, units_j = build_audio_units(script, iws)
    plan = InteractiveAudioPlan(script_index=script.index, units_i=units_i, units_j=units_j)
    refinement = refine_dialogue(script, build_llm(cfg, role="iar"), cfg)
    plan = refine_audio_plan(plan, refinement, cfg.iar.backchannel_reactivity)

    payload = {
        "script_index": plan.script_index,
        "sound_environment": dataclasses.asdict(plan.sound_environment) if plan.sound_environment else None,
        "units_i": [dataclasses.asdict(u) for u in plan.units_i],
        "units_j": [dataclasses.asdict(u) for u in plan.units_j],
    }
    with open(os.path.join(args.out, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    n_iw = sum(1 for u in plan.units_i + plan.units_j if u.kind == "interactive_word")
    print(f"saved plan.json (SE={plan.sound_environment.name}, interactive words={n_iw})")


if __name__ == "__main__":
    main()
