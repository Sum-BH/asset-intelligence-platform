# %% [markdown]
# # NB09 — Hybrid Retrieval + Evaluation (R15, R16)
# retrieve(query, k, filters) contract: dense (Qdrant) + BM25, fused by reciprocal-rank
# fusion. Eval set ≥40 Q/A; gate hit@5 ≥ 0.8 on hybrid. Conversion target: `aip_rag/retrieve.py`.

# %%
import json
import os
import re
from pathlib import Path
import yaml
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

DATA = Path("../../data")
COLL = "aip_knowledge"
url = os.environ.get("AIP_QDRANT_URL")
client = QdrantClient(url=url) if url else QdrantClient(path=str(DATA / "processed/qdrant_local"))

# corpus chunks (BM25 side) — rebuilt from source to stay self-contained
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
BACKEND = json.loads((DATA / "processed/chunking_report.json").read_text())["backend"]

def _embed(q):
    if BACKEND.startswith("ollama"):
        import requests
        r = requests.post("http://localhost:11434/api/embeddings",
                          json={"model": "nomic-embed-text", "prompt": q}, timeout=60)
        return r.json()["embedding"]
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=768, norm="l2", alternate_sign=False,
                           ngram_range=(1, 2), stop_words="english")
    return hv.transform([q]).toarray()[0].tolist()

# %% [markdown]
# ## retrieve() — the frozen contract

# %%
def retrieve(query: str, k: int = 5, filters: dict | None = None):
    dense = client.query_points(COLL, query=_embed(query), limit=20).points
    dense_rank = {p.payload["doc_id"]: r for r, p in enumerate(dense)}
    scores = bm25.get_scores(query.lower().split())
    kw = sorted(range(len(CH)), key=lambda i: -scores[i])[:20]
    kw_rank = {}
    for r, i in enumerate(kw):
        kw_rank.setdefault(CH[i]["doc_id"], r)
    # reciprocal-rank fusion at doc level, then best chunk per doc
    fused = {}
    for d, r in dense_rank.items():
        fused[d] = fused.get(d, 0) + 1 / (60 + r)
    for d, r in kw_rank.items():
        fused[d] = fused.get(d, 0) + 1 / (60 + r)
    top_docs = sorted(fused, key=fused.get, reverse=True)[:k]
    out = []
    for d in top_docs:
        best = max((i for i in range(len(CH)) if CH[i]["doc_id"] == d),
                   key=lambda i: scores[i])
        out.append({"chunk_id": CH[best]["chunk_id"], "doc_id": d,
                    "text": CH[best]["text"], "score": round(fused[d], 5)})
    return out

# %% [markdown]
# ## Eval set (40 Q/A) + metrics — gate: hybrid hit@5 ≥ 0.8

# %%
EVAL = [
 ("What does BPFO in the envelope spectrum indicate?", "heur-vib-0001"),
 ("How do I confirm stage-2 bearing damage?", "heur-vib-0001"),
 ("bearing outer race defect detection", "heur-vib-0001"),
 ("Why are envelope peaks slightly below theoretical frequency?", "heur-vib-0001"),
 ("How to tell imbalance from misalignment?", "heur-vib-0002"),
 ("high 2x vibration with axial component meaning", "heur-vib-0002"),
 ("what does soft foot mimic", "heur-vib-0002"),
 ("when should balancing be scheduled", "heur-vib-0002"),
 ("signs of pump cavitation in vibration", "heur-hyd-0003"),
 ("broadband noise rise suction crackling", "heur-hyd-0003"),
 ("required NPSH margin to avoid cavitation", "heur-hyd-0003"),
 ("recirculation vs cavitation confusion", "heur-hyd-0003"),
 ("ISO 4406 code jump meaning", "heur-hyd-0004"),
 ("filter differential pressure rising erratic valves", "heur-hyd-0004"),
 ("hydraulic oil sampling best practice", "heur-hyd-0004"),
 ("gear mesh frequency sidebands interpretation", "heur-gear-0005"),
 ("asymmetric sidebands around GMF", "heur-gear-0005"),
 ("when to inspect gearbox based on sidebands", "heur-gear-0005"),
 ("temperature creep with high frequency energy bearing", "heur-lub-0006"),
 ("how to confirm lubrication starvation", "heur-lub-0006"),
 ("over-greasing symptoms", "heur-lub-0006"),
 ("rotor bar fault current sidebands threshold", "heur-mcsa-0007"),
 ("detect mechanical faults from motor current", "heur-mcsa-0007"),
 ("VFD interference in current spectrum", "heur-mcsa-0007"),
 ("thermal delta T repair thresholds electrical", "heur-thermal-0008"),
 ("phase imbalance in thermal image", "heur-thermal-0008"),
 ("emissivity problem shiny connector thermal", "heur-thermal-0008"),
 ("minimum load for thermal inspection", "heur-thermal-0008"),
 ("how to set action window from RUL prediction", "heur-rul-0009"),
 ("RUL uncertainty planning RMSE", "heur-rul-0009"),
 ("RUL after partial repair reset", "heur-rul-0009"),
 ("accelerometer mounting bandwidth stud magnet", "heur-sensor-0010"),
 ("ski slope spectrum cause", "heur-sensor-0010"),
 ("validate new vibration alarm data quality", "heur-sensor-0010"),
 ("vibration severity zones A B C D", "std-iso10816-0011"),
 ("velocity RMS acceptable long term operation zone", "std-iso10816-0011"),
 ("escalation triggers from severity zones dashboard", "std-iso10816-0011"),
 ("ISO 4406 three number code particle sizes", "std-iso4406-0012"),
 ("cleanliness target servo hydraulic system", "std-iso4406-0012"),
 ("one code rise particle count doubling", "std-iso4406-0012"),
]
assert len(EVAL) >= 40

def metrics(fn, k=5):
    hits, rr = 0, 0.0
    for q, doc in EVAL:
        docs = [r["doc_id"] for r in fn(q, k=k)]
        if doc in docs:
            hits += 1
            rr += 1 / (docs.index(doc) + 1)
    return {"hit@5": round(hits / len(EVAL), 3), "mrr": round(rr / len(EVAL), 3)}

dense_only = lambda q, k=5: [
    {"doc_id": p.payload["doc_id"], "chunk_id": p.payload["chunk_id"],
     "text": p.payload["text"], "score": p.score}
    for p in client.query_points(COLL, query=_embed(q), limit=k).points]

m_hybrid, m_dense = metrics(retrieve), metrics(dense_only)
print("hybrid:", m_hybrid, "| dense-only:", m_dense, f"| backend: {BACKEND}")
assert m_hybrid["hit@5"] >= 0.8, "hybrid retrieval below gate"
json.dump({"hybrid": m_hybrid, "dense_only": m_dense, "backend": BACKEND, "n_eval": len(EVAL)},
          open(DATA / "processed/retrieval_metrics.json", "w"), indent=2)
(DATA / "processed/eval_set.jsonl").write_text(
    "\n".join(json.dumps({"question": q, "relevant_doc_ids": [d]}) for q, d in EVAL))
client.close()
print("NB09 complete.")
