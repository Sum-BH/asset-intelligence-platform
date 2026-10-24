"""Heading-aware chunking (promoted from NB08)."""

from __future__ import annotations

import re

DEFAULT_MAX_TOKENS = 650
DEFAULT_OVERLAP = 0.15


def chunk_doc(doc_id: str, body: str, max_tokens: int = DEFAULT_MAX_TOKENS,
              overlap: float = DEFAULT_OVERLAP) -> list[dict]:
    """Split on markdown headings first, then pack to ~max_tokens with overlap."""
    sections = re.split(r"(?=^## )", body, flags=re.M)
    chunks: list[tuple[str, str]] = []
    buf, head = "", ""
    n_tok = lambda s: len(s.split())  # noqa: E731
    for sec in sections:
        if sec.startswith("## "):
            head = sec.splitlines()[0]
        if n_tok(buf) + n_tok(sec) <= max_tokens:
            buf += sec
        else:
            if buf.strip():
                chunks.append((head, buf.strip()))
            tail = " ".join(buf.split()[-int(max_tokens * overlap):])
            buf = tail + "\n" + sec
    if buf.strip():
        chunks.append((head, buf.strip()))
    return [{"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "heading": h, "text": t}
            for i, (h, t) in enumerate(chunks)]


def chunk_corpus(docs: list[dict], **kw) -> list[dict]:
    out: list[dict] = []
    for d in docs:
        out += chunk_doc(d["doc_id"], d["body"], **kw)
    return out


def section_chunks(doc_id: str, body: str) -> list[dict]:
    """One chunk per markdown section — the retrieval granularity used in NB09+."""
    out = []
    for i, sec in enumerate(re.split(r"(?=^## )", body, flags=re.M)):
        if sec.strip():
            out.append({"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "text": sec.strip()})
    return out
