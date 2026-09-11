# CHAT (paper-aligned reimplementation)

Conversational Human Audio-visual Talking Dialogue Generation. This package
implements the paper's modules directly, with naming and notation kept in step
with the frozen camera-ready paper. For pipeline usage, weights, evaluation and
the current measured state, start at the repository [`README.md`](../README.md),
[`DOCS.md`](../DOCS.md) and [`DOCS.md`](../DOCS.md).

## Structure (maps to paper modules)

```
chat/
  schema.py      # Turn, IdentityDescriptor, DialogueScript (paper notation S^i / S^j)
  config.py      # dataclass config; configs/default.yaml documents the fields
  configs/       # default.yaml (no secrets, no machine paths)
  llm/           # LLMClient protocol + Gemini and OpenAI-compatible REST clients
  tdg/           # Textual Dialogue Generation            (paper Sec. 4.1)
  dadg/          # Dyadic Audio Dialogue Generation + IAR  (paper Sec. 4.1-4.2)
  ifbg/          # Interactive Facial Behaviour Gen + IFBR (paper Sec. 4.1, 4.3)
  eval/          # CSIM, Frechet (FID/FVD) and MCD metrics
```

## Configuration & secrets

No API keys or machine paths are stored in the repo. Set the key for whichever
provider you use in the environment:

```bash
export GEMINI_API_KEY=...     # package default, the paper's provider
export DEEPSEEK_API_KEY=...   # what scripts/x1_text.py defaults to
```

The variable name is configurable through `llm.api_key_env`. `chat.config` defaults
to Gemini, matching the paper; `scripts/x1_text.py` defaults to an OpenAI-compatible
endpoint (DeepSeek), which is what the reproduction runs used. See
[`DOCS.md`](../DOCS.md).

Load defaults or a custom YAML:

```python
from chat.config import Config
cfg = Config()                       # dataclass defaults
cfg = Config.load("my_config.yaml")  # override from YAML
```

## TDG usage

```python
from chat.config import Config
from chat.llm.gemini import GeminiClient
from chat.tdg import generate_dialogue_scripts

cfg = Config()
llm = GeminiClient(cfg.tdg.model, api_key_env=cfg.llm.api_key_env, endpoint=cfg.llm.endpoint)
scripts = generate_dialogue_scripts("Two friends catching up over coffee.", llm, cfg)
scripts[0].save("outputs/dg_1.json")
```

`generate_dialogue_scripts` builds each script segment by segment (5-10 turns per
segment, each conditioned on the previous), and returns `DialogueScript` objects
with paired identity descriptors `ID-txt^i` / `ID-txt^j`.

## Tests

```bash
python3 tests/test_tdg.py   # pure logic, no network/GPU
```
