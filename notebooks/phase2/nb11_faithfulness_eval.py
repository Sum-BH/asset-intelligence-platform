# %% [markdown]
# # NB11 — Faithfulness / Grounding Evaluation (R17 gate)
# Sandbox metric: lexical grounding — each cited sentence must share sufficient content
# tokens with its cited chunk. Workstation upgrade: RAGAS faithfulness with local LLM.
# Gate: mean faithfulness ≥ 0.85. Target: `aip_rag/eval.py`.

# %%
import json
import re
from pathlib import Path
import yaml

DATA = Path("../../data")
STOP = set("the a an of to in for is are and or with within on by as at from this that "
           "it its be not no least treat use using per here gives guidance indicators "
           "shows estimated".split())

def _stem(w):
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) > 3:
            w = w[: -len(suf)]
            break
    if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        w = w[:-1]
    return w

def content_tokens(s):
    s = re.sub(r"\[[\w#-]+\]", " ", s)          # citation markers are not claims
    return {_stem(w) for w in re.findall(r"[a-z]+", s.lower())
            if w not in STOP and len(w) > 2}

chunks = {}
for p in sorted((DATA / "corpus").glob("*.md")):
    fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", p.read_text(), re.S).groups()
    did = yaml.safe_load(fm)["doc_id"]
    for i, sec in enumerate(re.split(r"(?=^## )", body, flags=re.M)):
        if sec.strip():
            chunks[f"{did}#{i}"] = sec.strip()

# %%
def sentence_grounding(sent, chunk_text, evidence_text):
    """Grounded if the sentence's content tokens are covered by chunk ∪ evidence."""
    toks = content_tokens(sent)
    if not toks:
        return 1.0
    support = content_tokens(chunk_text) | content_tokens(evidence_text)
    return len(toks & support) / len(toks)

scores = {}
for rp in sorted((DATA / "processed/reports").glob("report_*.json")):
    rep = json.loads(rp.read_text())
    if rep.get("refused"):
        continue
    ev_text = json.dumps(rep["evidence"])
    per_sent = []
    for sec in ("evidence", "action", "uncertainty"):
        for s in re.split(r"(?<=[.!?]) ", rep["sections"][sec]):
            m = re.search(r"\[([\w#-]+)\]", s)
            if not m:
                continue
            per_sent.append(sentence_grounding(s, chunks.get(m.group(1), ""), ev_text))
    scores[rep["report_id"]] = round(sum(per_sent) / max(len(per_sent), 1), 3)

mean = sum(scores.values()) / max(len(scores), 1)
print("per-report grounding:", scores, "| mean:", round(mean, 3))
assert mean >= 0.85, "faithfulness below gate — inspect ungrounded sentences"
json.dump({"per_report": scores, "mean": round(mean, 3), "metric": "lexical-grounding-v1",
           "workstation_upgrade": "RAGAS faithfulness w/ local LLM"},
          open(DATA / "processed/faithfulness_scores.json", "w"), indent=2)
print("NB11 complete.")
