"""Corpus loading and validation (promoted from NB07)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

REQUIRED_FIELDS = {"doc_id", "title", "source_type", "tags", "author", "date"}
VALID_SOURCE_TYPES = {"authored", "standard_summary"}
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


class CorpusError(ValueError):
    """Raised when a corpus document violates the schema."""


def parse_doc(path: str | Path) -> dict:
    """Parse one corpus markdown file into a validated dict."""
    path = Path(path)
    text = path.read_text()
    m = _FRONTMATTER.match(text)
    if not m:
        raise CorpusError(f"{path.name}: missing YAML frontmatter")
    fm = yaml.safe_load(m.group(1))
    missing = REQUIRED_FIELDS - set(fm)
    if missing:
        raise CorpusError(f"{path.name}: missing fields {sorted(missing)}")
    if fm["source_type"] not in VALID_SOURCE_TYPES:
        raise CorpusError(f"{path.name}: invalid source_type {fm['source_type']!r}")
    if fm["doc_id"] != path.stem:
        raise CorpusError(f"{path.name}: doc_id {fm['doc_id']!r} != filename")
    return {**fm, "body": m.group(2).strip(),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def load_corpus(corpus_dir: str | Path) -> list[dict]:
    return [parse_doc(p) for p in sorted(Path(corpus_dir).glob("*.md"))]


def build_manifest(docs: list[dict]) -> dict:
    return {"n_docs": len(docs),
            "docs": [{"doc_id": d["doc_id"], "sha256": d["sha256"],
                      "source_type": d["source_type"], "tags": d["tags"]} for d in docs]}
