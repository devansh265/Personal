# Cast Resolver — Kuku AI Generalist Assignment, Problem 3

One script, one step, one checkable output: turn a folder of microdrama screenplays into a single cast list
so voice casting (TTS, then every dub language) is keyed on characters, not on whatever label the writer used that episode.

## Why
In the 50-episode input pack the same person is labelled many ways: `JULIANA` / `LADY JULIANA`, `GERALD` / `THE FORMER GROOM`,
`BLOND NOBLEMAN (Rupert)` / `RUPERT`. 64 speaker labels for roughly 45 speaking roles. Every label is a voice assignment,
once for the original and once per dub language (kukufm.com lists 7), so each stray label fans out.

## Run
```bash
python3 -I resolve_cast.py SCRIPTS_DIR raw_outputs                       # rules only, no key, free
python3 -I resolve_cast.py SCRIPTS_DIR raw_outputs --provider anthropic  # + one model call per show
# model is one function (llm_cluster) and one flag: --model claude-opus-5-5 / CAST_MODEL env
```
Input format: the pack's screenplay text (`**LABEL (tone):**` then `Start/End` then the line). Scene markers are matched case-insensitively,
because episode 6 writes them differently and a strict parser silently reads zero scenes.

## Output (`raw_outputs/`)
| file | what |
|---|---|
| `cast.json` | one row per character: canonical label, aliases, episodes, line count, and `drift_episodes` = an alias used *after* the character was already named |
| `alias_map.csv` | episode × raw label → canonical, with method (`identity`, `rule:honorific`, `rule:parenthetical`, `model`) and confidence |
| `flags.csv` | what a human must decide: unnamed roles with no naming evidence, low-confidence merges |
| `summary.txt` | the numbers |
| `llm_request.json` | exactly what the model is sent (not committed: it quotes the confidential scripts) |

## What runs where
- **Code**: parsing, counting, honorific strip (`LADY X` → `X` when `X` exists), parenthetical names (`BLOND NOBLEMAN (Rupert)` → `RUPERT`), drift detection.
- **Model** (optional, one call per show): clusters the remaining descriptors (`THE BEARDED STRANGER` → `ALEXANDER`) using the camera/action sentence before each line. Returns JSON with confidence and evidence.
- **Human**: anything under `--min-confidence` (0.8) or with no naming evidence stays a flag. `LORD` in ep 33 (36 lines, never named) is the canonical example: flag, never merge.

## Honest limits
- The committed `raw_outputs/` are from the rules-only run (no API key on the machine at submission time). Rules alone resolve 3 aliases and surface 26 flags; the model path is written and swappable but unrun on this pack.
- Deliberate double identities (disguise, twins) would be merged by the model unless the context says otherwise.
- One-line extras are low confidence by design, so the review list grows with cast size.
