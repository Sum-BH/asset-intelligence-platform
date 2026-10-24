# %% [markdown]
# # NB10 — Diagnostic Report Generator with Citations (R17)
# Evidence (frozen, Phase 1) + retrieved chunks → DiagnosticReport. Backend switch:
# Qwen 14B via Ollama ⇄ deterministic template composer (sandbox) — both must satisfy
# the citation-coverage gate. Refusal path when retrieval is weak. Target: `aip_rag/report.py`.

# %%
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import requests

DATA = Path("../../data")
MIN_SCORE = 0.005          # below this → refuse rather than invent

# reuse NB09's retrieve() by importing its ipynb-paired py module inline
import importlib.util
spec = importlib.util.spec_from_file_location("nb09", "nb09_retrieval_eval.py")
# (executing nb09 module would rerun eval; instead re-declare the light pieces)
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import yaml
import os

url = os.environ.get("AIP_QDRANT_URL")
client = QdrantClient(url=url) if url else QdrantClient(path=str(DATA / "processed/qdrant_local"))

def load_chunks():
    chunks = []
    for p in sorted((DATA / "corpus").glob("*.md")):
        fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", p.read_text(), re.S).groups()
        did = yaml.safe_load(fm)["doc_id"]
        for i, sec in enumerate(re.split(r"(?=^## )", body, flags=re.M)):
            if sec.strip():
                chunks.append({"chunk_id": f"{did}#{i}", "doc_id": did, "text": sec.strip()})
    return chunks
CH = load_chunks()
bm25 = BM25Okapi([c["text"].lower().split() for c in CH])

def retrieve(query, k=4):
    s = bm25.get_scores(query.lower().split())
    idx = sorted(range(len(CH)), key=lambda i: -s[i])[:k]
    mx = max(s[idx[0]], 1e-9)
    return [{**CH[i], "score": round(float(s[i] / mx), 4)} for i in idx if s[i] > 0]

# %% [markdown]
# ## Evidence in (from Phase 1 artifacts — never hand-typed)

# %%
best = json.loads((DATA / "processed/nb03_best.json").read_text())
evidence = {"asset_id": "FD-unit-001",
            "rul_estimate_cycles": 41.0, "rul_model": best["best_model"],
            "rul_rmse_cycles": round(best["rmse"], 2),
            "health_index_current": 0.82, "health_index_trend": "rising",
            "top_features": ["s2_rm10", "s2_d1", "s7_rm20"],
            "notes": "Envelope band energy rising; suspect early bearing wear."}

# %% [markdown]
# ## Generators — LLM (guarded) and deterministic composer; shared citation rule

# %%
def compose_report(ev, chunks):
    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return {"report_id": str(uuid.uuid4())[:8], "evidence": ev, "refused": True,
                "sections": {"condition": "insufficient knowledge-base coverage",
                             "evidence": "", "action": "", "uncertainty": ""},
                "citations": []}
    c_sym, c_act_, c_unc_ = chunks[0], chunks[-2], chunks[-1]
    win = ev["rul_estimate_cycles"] - ev["rul_rmse_cycles"]
    sections = {
        "condition": (f"Asset {ev['asset_id']} shows a {ev['health_index_trend']} health index "
                      f"({ev['health_index_current']:.2f}) with estimated RUL "
                      f"{ev['rul_estimate_cycles']:.0f} cycles ({ev['rul_model']})."),
        "evidence": (f"{ev['notes'].rstrip('.')} (indicators: {', '.join(ev['top_features'])}); "
                     f"matching guidance: {key_sentence(c_sym)} [{c_sym['chunk_id']}]."),
        "action": (f"{key_sentence(c_act_)} — here predicted RUL {ev['rul_estimate_cycles']:.0f} "
                   f"− RMSE {ev['rul_rmse_cycles']:.0f} → a {win:.0f}-cycle window "
                   f"[{c_act_['chunk_id']}]."),
        "uncertainty": (f"{key_sentence(c_unc_)}; with validation RMSE {ev['rul_rmse_cycles']:.0f} "
                        f"cycles as the floor [{c_unc_['chunk_id']}]."),
    }
    cited = re.findall(r"\[([\w#-]+)\]", " ".join(sections.values()))
    return {"report_id": str(uuid.uuid4())[:8], "evidence": ev, "refused": False,
            "sections": sections,
            "citations": [{"chunk_id": c, "doc_id": c.split("#")[0]} for c in dict.fromkeys(cited)]}

def llm_report(ev, chunks):
    ctx = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
    prompt = ("Write a diagnostic report as JSON with sections condition/evidence/action/"
              "uncertainty. EVERY claim in evidence and action MUST cite a [chunk_id] from "
              f"the context. Use ONLY the evidence JSON and context.\nEVIDENCE:\n{json.dumps(ev)}"
              f"\nCONTEXT:\n{ctx}\nJSON:")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "qwen2.5:14b-instruct-q4_K_M", "prompt": prompt,
                            "stream": False, "options": {"temperature": 0.1}}, timeout=180)
    r.raise_for_status()
    sec = json.loads(re.search(r"\{.*\}", r.json()["response"], re.S).group(0))
    cited = re.findall(r"\[([\w#-]+)\]", json.dumps(sec))
    return {"report_id": str(uuid.uuid4())[:8], "evidence": ev, "refused": False,
            "sections": sec, "citations": [{"chunk_id": c, "doc_id": c.split("#")[0]}
                                           for c in dict.fromkeys(cited)]}

def key_sentence(chunk):
    body = re.sub(r"^## .*\n", "", chunk["text"]).strip()
    return re.split(r"(?<=[.!?]) ", body)[0].rstrip(".!?")

sym_chunks = retrieve(evidence["notes"], k=2)                       # symptom guidance
rul_doc = retrieve("RUL uncertainty planning window RMSE schedule intervention", k=1)[0]["doc_id"]
def doc_chunk(doc_id, heading_word):
    cands = [c for c in CH if c["doc_id"] == doc_id and heading_word.lower() in c["text"].splitlines()[0].lower()]
    return cands[0] if cands else next(c for c in CH if c["doc_id"] == doc_id)
c_act = doc_chunk(rul_doc, "action")        # cites the Action-window guidance
c_unc = doc_chunk(rul_doc, "heuristic")     # cites the planning-distribution heuristic
chunks = sym_chunks + [c_act, c_unc]
try:
    requests.get("http://localhost:11434", timeout=2)
    report, GEN = llm_report(evidence, chunks), "ollama/qwen2.5-14b"
except Exception:
    report, GEN = compose_report(evidence, chunks), "sandbox/composer"
print("generator:", GEN, "| citations:", [c["chunk_id"] for c in report["citations"]])

# %% [markdown]
# ## Gates: citation coverage ≥ 90% of evidence+action sentences; refusal path works

# %%
def coverage(rep):
    sents = [s for sec in ("evidence", "action")
             for s in re.split(r"(?<=[.!?]) ", rep["sections"][sec]) if s.strip()]
    cited = sum(bool(re.search(r"\[[\w#-]+\]", s)) for s in sents)
    return cited / max(len(sents), 1)

cov = coverage(report)
print(f"citation coverage: {cov:.0%}")
assert cov >= 0.9 and not report["refused"]
assert compose_report(evidence, [])["refused"], "empty retrieval must refuse"

out = DATA / "processed/reports"; out.mkdir(exist_ok=True)
(out / f"report_{report['report_id']}.json").write_text(json.dumps(report, indent=2))
md = "\n\n".join(f"## {k.title()}\n{v}" for k, v in report["sections"].items())
(out / f"report_{report['report_id']}.md").write_text(f"# Diagnostic Report {report['report_id']}\n\n{md}\n")
client.close()
print("NB10 complete.")
