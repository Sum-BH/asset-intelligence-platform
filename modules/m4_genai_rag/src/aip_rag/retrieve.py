"""Retrieval over the local knowledge base (promoted from NB09/NB10).

BM25Retriever is the dependency-light path used everywhere in-repo. On the
workstation, HybridRetriever adds dense Qdrant results fused by reciprocal rank.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from aip_rag.chunking import section_chunks
from aip_rag.corpus import load_corpus

RRF_K = 60


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "text": self.text, "score": self.score}


class BM25Retriever:
    def __init__(self, corpus_dir):
        docs = load_corpus(corpus_dir)
        self.chunks: list[dict] = []
        for d in docs:
            self.chunks += section_chunks(d["doc_id"], d["body"])
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._bm25.get_scores(query.lower().split())
        idx = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [Hit(self.chunks[i]["chunk_id"], self.chunks[i]["doc_id"],
                    self.chunks[i]["text"], round(float(scores[i]), 4))
                for i in idx if scores[i] > 0]

    def doc_chunk(self, doc_id: str, heading_word: str) -> dict | None:
        """Pick the section of a document whose heading matches (NB10's rule)."""
        for c in self.chunks:
            if c["doc_id"] == doc_id and heading_word.lower() in c["text"].splitlines()[0].lower():
                return c
        return next((c for c in self.chunks if c["doc_id"] == doc_id), None)


def reciprocal_rank_fusion(*rankings: list[str], k: int = RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def evidence_strength(top_score: float, strong: float = 3.0, weak: float = 0.5) -> str:
    return "strong" if top_score >= strong else ("weak" if top_score > weak else "none")
