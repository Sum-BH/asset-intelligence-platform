# %% [markdown]
# # NB08 — Chunking, Embeddings & Qdrant Index (R13, R14)
# Backend switch: Ollama nomic-embed-text (workstation) ⇄ deterministic hashing
# embeddings (sandbox). Qdrant: server via AIP_QDRANT_URL ⇄ local-file mode.
# Idempotent by doc hash. Conversion target: `aip_rag/{chunking,index}.py`.

# %%
import json
import os
import re
from pathlib import Path
import requests
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DATA = Path("../../data")
DIM = 768
CHUNK_TOK, OVERLAP = 650, 0.15

# %% [markdown]
# ## 1. Embedding backend

# %%
def _ollama_embed(texts):
    out = []
    for t in texts:
        r = requests.post("http://localhost:11434/api/embeddings",
                          json={"model": "nomic-embed-text", "prompt": t}, timeout=60)
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out

def _hash_embed(texts):
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=DIM, norm="l2", alternate_sign=False,
                           ngram_range=(1, 2), stop_words="english")
    return hv.transform(texts).toarray().tolist()

try:
    requests.get("http://localhost:11434", timeout=2)
    embed, BACKEND = _ollama_embed, "ollama/nomic-embed-text"
except Exception:
    embed, BACKEND = _hash_embed, "sandbox/hashing-768"
print("embedding backend:", BACKEND)

# %% [markdown]
# ## 2. Heading-aware chunking (~650 tokens, 15% overlap)

# %%
def chunk_doc(doc_id: str, body: str):
    sections = re.split(r"(?=^## )", body, flags=re.M)
    chunks, buf, head = [], "", ""
    tok = lambda s: len(s.split())
    for sec in sections:
        if sec.startswith("## "):
            head = sec.splitlines()[0]
        if tok(buf) + tok(sec) <= CHUNK_TOK:
            buf += sec
        else:
            if buf.strip():
                chunks.append((head, buf.strip()))
            tail = " ".join(buf.split()[-int(CHUNK_TOK * OVERLAP):])
            buf = tail + "\n" + sec
    if buf.strip():
        chunks.append((head, buf.strip()))
    return [{"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id,
             "heading": h, "text": t} for i, (h, t) in enumerate(chunks)]

manifest = json.loads((DATA / "processed/corpus_manifest.json").read_text())
meta = {d["doc_id"]: d for d in manifest["docs"]}
all_chunks = []
for p in sorted((DATA / "corpus").glob("*.md")):
    fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", p.read_text(), re.S).groups()
    all_chunks += chunk_doc(yaml.safe_load(fm)["doc_id"], body)
assert all(len(c["text"].split()) <= CHUNK_TOK + 50 for c in all_chunks)
print(f"{len(all_chunks)} chunks from {manifest['n_docs']} docs")

# %% [markdown]
# ## 3. Qdrant upsert (idempotent by doc hash)

# %%
url = os.environ.get("AIP_QDRANT_URL")
client = QdrantClient(url=url) if url else QdrantClient(path=str(DATA / "processed/qdrant_local"))
COLL = "aip_knowledge"
state_f = DATA / "processed/index_state.json"
prev = json.loads(state_f.read_text()) if state_f.exists() else {}
changed = [d for d in manifest["docs"] if prev.get(d["doc_id"]) != d["sha256"]]

if COLL not in [c.name for c in client.get_collections().collections]:
    client.create_collection(COLL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

if changed:
    todo = [c for c in all_chunks if c["doc_id"] in {d["doc_id"] for d in changed}]
    vecs = embed([c["heading"] + "\n" + c["text"] for c in todo])
    pts = [PointStruct(id=abs(hash(c["chunk_id"])) % (10**12), vector=v,
                       payload={**c, "tags": meta[c["doc_id"]]["tags"],
                                "source_type": meta[c["doc_id"]]["source_type"],
                                "embed_backend": BACKEND})
           for c, v in zip(todo, vecs)]
    client.upsert(COLL, pts)
    print(f"upserted {len(pts)} chunks ({len(changed)} changed docs)")
else:
    print("upserted 0 — corpus unchanged (idempotency OK)")
state_f.write_text(json.dumps({d["doc_id"]: d["sha256"] for d in manifest["docs"]}))

n = client.count(COLL).count
assert n == len(all_chunks), f"collection {n} != chunks {len(all_chunks)}"
json.dump({"n_chunks": len(all_chunks), "backend": BACKEND, "dim": DIM},
          open(DATA / "processed/chunking_report.json", "w"))
client.close()
print("NB08 complete.")
