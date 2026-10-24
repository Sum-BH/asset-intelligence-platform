"""Ported from NB07-NB11 in-notebook asserts."""
import pytest
from aip_core import paths
from aip_rag.chunking import chunk_doc, section_chunks
from aip_rag.corpus import CorpusError, load_corpus, parse_doc
from aip_rag.report import citation_coverage, compose_report, grounding_score
from aip_rag.retrieve import BM25Retriever, reciprocal_rank_fusion

CORPUS = paths.data / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")


def test_corpus_validates():
    docs = load_corpus(CORPUS)
    assert len(docs) >= 10
    assert all(d["source_type"] in {"authored", "standard_summary"} for d in docs)


def test_corpus_rejects_bad_frontmatter(tmp_path):
    p = tmp_path / "bad-doc.md"
    p.write_text("no frontmatter here")
    with pytest.raises(CorpusError):
        parse_doc(p)


def test_chunking_respects_size():
    docs = load_corpus(CORPUS)
    chunks = chunk_doc(docs[0]["doc_id"], docs[0]["body"], max_tokens=200)
    assert chunks and all(len(c["text"].split()) <= 250 for c in chunks)
    assert all(c["chunk_id"].startswith(docs[0]["doc_id"] + "#") for c in chunks)


def test_section_chunks_ids_unique():
    docs = load_corpus(CORPUS)
    ids = [c["chunk_id"] for d in docs for c in section_chunks(d["doc_id"], d["body"])]
    assert len(ids) == len(set(ids))


def test_retrieval_finds_bearing_doc():
    r = BM25Retriever(CORPUS)
    hits = r.search("bearing envelope BPFO outer race defect", k=5)
    assert hits and hits[0].doc_id.startswith("heur-vib")


def test_retrieval_empty_on_nonsense():
    r = BM25Retriever(CORPUS)
    assert r.search("zzzz qqqq xxxx") == []


def test_rrf_prefers_consensus():
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["b", "a", "d"])
    assert fused["b"] > fused["c"] and fused["a"] > fused["d"]


@pytest.fixture
def evidence():
    return {"asset_id": "A-1", "rul_estimate_cycles": 41.0, "rul_model": "gb",
            "rul_rmse_cycles": 9.0, "health_index_current": 0.82,
            "health_index_trend": "rising", "top_features": ["s2_rm10"],
            "notes": "Envelope band energy rising"}


def test_report_cites_every_claim(evidence):
    r = BM25Retriever(CORPUS)
    sym = [c for c in (h.as_dict() for h in r.search("bearing envelope", k=2))]
    rul_doc = r.search("RUL uncertainty planning window RMSE", k=1)[0].doc_id
    chunks = sym + [r.doc_chunk(rul_doc, "action"), r.doc_chunk(rul_doc, "heuristic")]
    rep = compose_report(evidence, chunks)
    assert not rep["refused"]
    assert citation_coverage(rep) >= 0.9
    assert rep["citations"]


def test_report_refuses_without_evidence(evidence):
    assert compose_report(evidence, [])["refused"] is True


def test_grounding_score_detects_unsupported():
    good = grounding_score("bearing envelope spectrum peaks",
                           "bearing envelope spectrum shows peaks", "")
    bad = grounding_score("quantum flux capacitor anomaly", "bearing envelope", "")
    assert good >= 0.85 > bad
