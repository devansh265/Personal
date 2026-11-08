#!/usr/bin/env python3
"""
resolve_cast.py - turn a folder of microdrama screenplays into one cast list.

Problem: the same character is labelled many ways across episodes
(JULIANA / LADY JULIANA, GERALD / THE FORMER GROOM, BLOND NOBLEMAN (Rupert) / RUPERT).
Every label becomes a voice assignment, once for TTS and once per dub language,
so a 65-label cast for ~45 people means wrong or swapped voices.

What this does
  1. code   : parse every script, count labels, lines, episodes, parenthetical hints,
              and the CAMERA/ACTION sentence just before each label (context).
  2. rules  : merge the cheap, certain cases (honorific strip, parenthetical names).
  3. model  : optional - one call per show that clusters the remaining labels using
              the context snippets. Swappable via --provider / CAST_MODEL.
  4. human  : anything below --min-confidence or with no naming evidence is a flag,
              never a merge.

Outputs (in OUT_DIR)
  cast.json        one row per character: id, aliases, episodes, lines, drift episodes
  alias_map.csv    episode, raw label, canonical, method, confidence
  flags.csv        what a human must decide, with the evidence
  summary.txt      the numbers for the doc
  llm_request.json / llm_response.json   raw model I/O when --provider != none

Usage
  python3 -I resolve_cast.py SCRIPTS_DIR OUT_DIR [--provider none|anthropic]
                                               [--model claude-opus-5-5]
                                               [--min-confidence 0.8]
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

NON_SPEAKERS = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TITLE", "TEXT", "VO", "V.O.", "NARRATOR VO"}
HONORIFICS = ("LADY ", "LORD ", "SIR ", "THE ", "MISS ", "MRS ", "MR ", "DUKE ", "DUCHESS ")
# words that mean "a role, not a name" - a label made only of these is unnamed
ROLE_WORDS = set("""LORD LADY MAID GUARD GUARDS NOBLEMAN NOBLEWOMAN MAN WOMAN BOY GIRL THUG THUGS PRIEST JUDGE
CONSTABLE OFFICIAL SOLDIER SENTRY SCOUT FIGHTER DRIVER BUTLER FOOTMAN HOUSEKEEPER INFORMANT VILLAGER TOWNSMAN
MOTHER FATHER STRANGER GROOM BRIDE MATRIARCH ELDER YOUNG OLD TALL BALD BLOND BEARDED SCARRED MYSTERIOUS
FORMER NOBLE ROYAL CROWN HIGH POLICE PEASANT ARISTOCRATIC CART LOYALIST CONSPIRATOR OFF-SCREEN VOICE
WITH CAP GOLD GOWN IN THIRD SECOND FIRST ALEXANDER'S DOWAGER DUCHESS""".split())

LABEL_RE = re.compile(
    r"^\*\*([A-Z][A-Z0-9 '\-\.]+?)((?:\s*\([^)]*\))*):\*\*\s*\n"
    r"Start: \((\d+):(\d+\.\d+)\) \| End: \((\d+):(\d+\.\d+)\)\s*\n"
    r"(.*)$",
    re.M,
)
SCENE_RE = re.compile(r"^scene end:", re.I | re.M)
EP_RE = re.compile(r"episode[-_ ]0*(\d+)", re.I)


# ----------------------------------------------------------------------------- 1. parse
def parse_folder(folder):
    """Return (labels, episodes) where labels[name] holds counts, eps, hints, contexts."""
    labels = collections.defaultdict(lambda: {"lines": 0, "eps": set(), "hints": set(), "ctx": [], "first_ep": 999})
    episodes = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        text = open(path, encoding="utf-8", errors="ignore").read()
        m = EP_RE.search(os.path.basename(path)) or re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text)
        ep = int(m.group(1)) if m else len(episodes) + 1
        scenes = len(SCENE_RE.findall(text))
        episodes.append({"ep": ep, "file": os.path.basename(path), "scenes": scenes})
        last_desc = ""
        for mm in LABEL_RE.finditer(text):
            name, body = mm.group(1).strip(), mm.group(7).strip()
            parens = re.findall(r"\(([^)]*)\)", mm.group(2) or "")
            paren = parens[0] if parens else ""  # "(Rupert) (Cold)" -> Rupert
            if name in ("CAMERA", "ACTION"):
                last_desc = body
                continue
            if name in NON_SPEAKERS:
                continue
            rec = labels[name]
            rec["lines"] += 1
            rec["eps"].add(ep)
            rec["first_ep"] = min(rec["first_ep"], ep)
            # a parenthetical that carries a proper name: "(Rupert, Cold)" -> Rupert
            # only the FIRST comma-separated token, only if it is a lone proper name: "(Rupert, Cold)" yes, "(To Clarissa)" no
            first = re.match(r"\s*([A-Z][a-z]{2,})\b", paren)
            if first and first.group(1).upper() not in ROLE_WORDS:
                rec["hints"].add(first.group(1).upper())
            if last_desc and len(rec["ctx"]) < 4:
                rec["ctx"].append("ep%02d: %s" % (ep, last_desc[:160]))
    return labels, episodes


# ----------------------------------------------------------------------------- 2. rules
def strip_honorific(label):
    for h in HONORIFICS:
        if label.startswith(h):
            return label[len(h):].strip()
    return label


def is_unnamed(label):
    return all(w in ROLE_WORDS for w in label.replace("'S", "").split())


def rule_merges(labels):
    """Certain merges only. Returns {alias: (canonical, method)}."""
    merges = {}
    names = set(labels)
    for lab in names:
        core = strip_honorific(lab)
        if core != lab and core in names and not is_unnamed(core):
            merges[lab] = (core, "rule:honorific")
    for lab, rec in labels.items():
        for hint in rec["hints"]:
            if hint in names and hint != lab:
                merges[lab] = (hint, "rule:parenthetical")
    return merges


# ----------------------------------------------------------------------------- 3. model
def build_prompt(labels, merges):
    rows = []
    for lab, rec in sorted(labels.items(), key=lambda kv: -kv[1]["lines"]):
        rows.append({
            "label": lab, "lines": rec["lines"], "episodes": sorted(rec["eps"]),
            "already_merged_into": merges.get(lab, (None,))[0],
            "context": rec["ctx"][:3],
        })
    instructions = (
        "You are the continuity editor for a microdrama. Below are every speaker label in one show, "
        "with line counts, episodes and the camera/action sentence that preceded a line. Group labels that "
        "are the same character. Rules: a descriptor used BEFORE a character is named (e.g. 'THE BEARDED STRANGER' "
        "then 'ALEXANDER') is the same person if the context supports it. Never merge two labels that speak in the "
        "same scene to each other. If a label has no naming evidence, leave it alone and say so. "
        "Return ONLY JSON: {\"clusters\":[{\"canonical\":str,\"aliases\":[str],\"confidence\":0-1,\"evidence\":str}],"
        "\"unresolved\":[{\"label\":str,\"reason\":str}]}"
    )
    return {"instructions": instructions, "labels": rows}


def llm_cluster(payload, provider, model):
    """The one swappable function. Add a provider here; nothing else changes."""
    if provider == "none":
        return None
    if provider == "anthropic":
        import anthropic  # official SDK; needs ANTHROPIC_API_KEY or an `ant auth login` profile
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=16000,
            system=payload["instructions"],
            messages=[{"role": "user", "content": json.dumps(payload["labels"], ensure_ascii=False)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        text = text[text.find("{"): text.rfind("}") + 1]
        return json.loads(text)
    raise SystemExit("unknown provider %r (none|anthropic)" % provider)


# ----------------------------------------------------------------------------- 4. assemble
def assemble(labels, merges, llm, min_conf):
    canon_of = {lab: lab for lab in labels}
    method = {lab: "identity" for lab in labels}
    conf = {lab: 1.0 for lab in labels}
    flags = []
    for alias, (canon, how) in merges.items():
        canon_of[alias], method[alias], conf[alias] = canon, how, 0.95
    if llm:
        for cl in llm.get("clusters", []):
            canon = cl["canonical"].upper().strip()
            for alias in cl.get("aliases", []):
                alias = alias.upper().strip()
                if alias not in labels or alias == canon:
                    continue
                if cl.get("confidence", 0) >= min_conf:
                    canon_of[alias], method[alias], conf[alias] = canon, "model", cl["confidence"]
                else:
                    flags.append((alias, "low-confidence merge -> %s" % canon, cl.get("evidence", ""), cl.get("confidence", 0)))
        for u in llm.get("unresolved", []):
            flags.append((u["label"], "model: unresolved", u.get("reason", ""), 0))
    # resolve chains (A->B->C)
    for lab in canon_of:
        seen = set()
        while canon_of[lab] != canon_of[canon_of[lab]] and canon_of[lab] not in seen:
            seen.add(canon_of[lab]); canon_of[lab] = canon_of[canon_of[lab]]
    # unnamed roles that nobody merged
    for lab, rec in labels.items():
        if canon_of[lab] == lab and is_unnamed(lab) and rec["lines"] >= 3 and not any(f[0] == lab for f in flags):
            flags.append((lab, "unnamed role, no naming evidence", "%d lines, eps %s" % (rec["lines"], sorted(rec["eps"])), 0))
    # cast rows
    cast = {}
    for lab, rec in labels.items():
        c = canon_of[lab]
        row = cast.setdefault(c, {"id": c.title(), "canonical_label": c, "aliases": [], "episodes": set(), "lines": 0, "drift_episodes": []})
        row["lines"] += rec["lines"]; row["episodes"] |= rec["eps"]
        if lab != c:
            row["aliases"].append(lab)
    # drift: an alias used in an episode AFTER the canonical label first appeared
    for c, row in cast.items():
        named_at = labels[c]["first_ep"] if c in labels else 999
        for a in row["aliases"]:
            late = sorted(e for e in labels[a]["eps"] if e > named_at)
            if late:
                row["drift_episodes"].append({"alias": a, "named_in_ep": named_at, "alias_recurs_in": late})
        row["episodes"] = sorted(row["episodes"])
    cast_rows = sorted(cast.values(), key=lambda r: -r["lines"])
    return cast_rows, canon_of, method, conf, flags


def write_outputs(out, labels, episodes, cast_rows, canon_of, method, conf, flags, payload, llm):
    os.makedirs(out, exist_ok=True)
    json.dump(cast_rows, open(os.path.join(out, "cast.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "alias_map.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "label", "canonical", "method", "confidence", "lines_in_ep"])
        for lab, rec in sorted(labels.items()):
            for ep in sorted(rec["eps"]):
                w.writerow([ep, lab, canon_of[lab], method[lab], conf[lab], ""])
    with open(os.path.join(out, "flags.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "issue", "evidence", "confidence"])
        for row in sorted(flags): w.writerow(row)
    bad = [e for e in episodes if e["scenes"] == 0]
    total_lines = sum(r["lines"] for r in labels.values())
    leads = [r for r in cast_rows if r["lines"] >= 20]
    lead_labels = sum(1 + len(r["aliases"]) for r in leads)
    drift = sum(len(r["drift_episodes"]) for r in cast_rows)
    summary = [
        "episodes parsed: %d (%d with zero scene markers: %s)" % (len(episodes), len(bad), ", ".join(e["file"] for e in bad) or "-"),
        "dialogue lines: %d" % total_lines,
        "speaker labels: %d" % len(labels),
        "characters after resolve: %d" % len(cast_rows),
        "leads (>=20 lines): %d carrying %d labels" % (len(leads), lead_labels),
        "aliases that recur after the character was named (drift): %d" % drift,
        "flags for human review: %d" % len(flags),
        "model: %s" % ("none (rules only)" if not llm else "used"),
    ]
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(summary) + "\n")
    json.dump(payload, open(os.path.join(out, "llm_request.json"), "w"), indent=1, ensure_ascii=False)
    if llm is not None:
        json.dump(llm, open(os.path.join(out, "llm_response.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts_dir"); ap.add_argument("out_dir")
    ap.add_argument("--provider", default=os.environ.get("CAST_PROVIDER", "none"), help="none | anthropic")
    ap.add_argument("--model", default=os.environ.get("CAST_MODEL", "claude-opus-5-5"))
    ap.add_argument("--min-confidence", type=float, default=0.8)
    a = ap.parse_args()
    labels, episodes = parse_folder(a.scripts_dir)
    if not labels:
        sys.exit("no speaker labels found in %s" % a.scripts_dir)
    merges = rule_merges(labels)
    payload = build_prompt(labels, merges)
    llm = llm_cluster(payload, a.provider, a.model)
    cast_rows, canon_of, method, conf, flags = assemble(labels, merges, llm, a.min_confidence)
    write_outputs(a.out_dir, labels, episodes, cast_rows, canon_of, method, conf, flags, payload, llm)


if __name__ == "__main__":
    main()
