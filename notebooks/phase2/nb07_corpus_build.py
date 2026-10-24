# %% [markdown]
# # NB07 — Corpus Build & Validation (R12)
# Validates every corpus doc against the frontmatter schema, builds the manifest
# with content hashes (idempotent re-index key for NB08). Conversion target: `aip_rag/corpus.py`.

# %%
import hashlib
import json
import re
from pathlib import Path
import yaml

DATA = Path("../../data")
CORPUS = DATA / "corpus"
REQUIRED = {"doc_id", "title", "source_type", "tags", "author", "date"}
VALID_TYPES = {"authored", "standard_summary"}

# %%
def parse_doc(path: Path) -> dict:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert m, f"{path.name}: missing frontmatter"
    fm = yaml.safe_load(m.group(1))
    missing = REQUIRED - set(fm)
    assert not missing, f"{path.name}: missing fields {missing}"
    assert fm["source_type"] in VALID_TYPES, f"{path.name}: bad source_type"
    assert fm["doc_id"] == path.stem, f"{path.name}: doc_id != filename"
    return {**fm, "body": m.group(2).strip(),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}

docs = [parse_doc(p) for p in sorted(CORPUS.glob("*.md"))]
authored = sum(d["source_type"] == "authored" for d in docs)
print(f"{len(docs)} docs validated ({authored} authored, {len(docs)-authored} standard summaries)")
assert len(docs) >= 10, "corpus too small — write more heuristics before indexing"

# %% [markdown]
# ## Manifest (drives NB08's idempotent indexing)

# %%
manifest = {"generated": "nb07", "n_docs": len(docs),
            "docs": [{"doc_id": d["doc_id"], "sha256": d["sha256"],
                      "source_type": d["source_type"], "tags": d["tags"]} for d in docs]}
(DATA / "processed/corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
drafts = sum("DRAFT" in str(d.get("review_status", "")) for d in docs)
print(f"manifest written; NOTE: {drafts} docs are starter DRAFTS — rewrite before publishing")
print("NB07 complete.")
