"""Cited diagnostic report generation (promoted from NB10/NB11)."""

from __future__ import annotations

import re
import uuid

MIN_SCORE = 0.005
_CITATION = re.compile(r"\[([\w#-]+)\]")


def key_sentence(chunk: dict) -> str:
    body = re.sub(r"^## .*\n", "", chunk["text"]).strip()
    return re.split(r"(?<=[.!?]) ", body)[0].rstrip(".!?")


def compose_report(evidence: dict, chunks: list[dict]) -> dict:
    """Deterministic composer. Each section is ONE cited sentence — the structure that
    satisfies the citation-coverage gate (learned the hard way in NB10)."""
    report_id = str(uuid.uuid4())[:8]
    if not chunks or chunks[0].get("score", 1.0) < MIN_SCORE:
        return {"report_id": report_id, "evidence": evidence, "refused": True,
                "sections": {"condition": "insufficient knowledge-base coverage",
                             "evidence": "", "action": "", "uncertainty": ""},
                "citations": []}
    c_sym, c_act, c_unc = chunks[0], chunks[-2], chunks[-1]
    win = evidence["rul_estimate_cycles"] - evidence["rul_rmse_cycles"]
    sections = {
        "condition": (f"Asset {evidence['asset_id']} shows a {evidence['health_index_trend']} "
                      f"health index ({evidence['health_index_current']:.2f}) with estimated "
                      f"RUL {evidence['rul_estimate_cycles']:.0f} cycles "
                      f"({evidence['rul_model']})."),
        "evidence": (f"{evidence['notes'].rstrip('.')} "
                     f"(indicators: {', '.join(evidence['top_features'])}); matching "
                     f"guidance: {key_sentence(c_sym)} [{c_sym['chunk_id']}]."),
        "action": (f"{key_sentence(c_act)} — here predicted RUL "
                   f"{evidence['rul_estimate_cycles']:.0f} − RMSE "
                   f"{evidence['rul_rmse_cycles']:.0f} gives a {win:.0f}-cycle window "
                   f"[{c_act['chunk_id']}]."),
        "uncertainty": (f"Treat RUL as a planning distribution, not a date; the RMSE of "
                        f"{evidence['rul_rmse_cycles']:.0f} cycles is the uncertainty floor "
                        f"[{c_unc['chunk_id']}]."),
    }
    cited = list(dict.fromkeys(_CITATION.findall(" ".join(sections.values()))))
    return {"report_id": report_id, "evidence": evidence, "refused": False,
            "sections": sections,
            "citations": [{"chunk_id": c, "doc_id": c.split("#")[0]} for c in cited]}


def citation_coverage(report: dict, sections=("evidence", "action")) -> float:
    sents = [s for sec in sections
             for s in re.split(r"(?<=[.!?]) ", report["sections"][sec]) if s.strip()]
    if not sents:
        return 0.0
    return sum(bool(_CITATION.search(s)) for s in sents) / len(sents)


_STOP = set("the a an of to in for is are and or with within on by as at from this that "
            "it its be not no least treat use using per here gives guidance indicators "
            "shows estimated".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) > 3:
            w = w[: -len(suf)]
            break
    if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        w = w[:-1]
    return w


def content_tokens(s: str) -> set[str]:
    s = _CITATION.sub(" ", s)
    return {_stem(w) for w in re.findall(r"[a-z]+", s.lower())
            if w not in _STOP and len(w) > 2}


def grounding_score(sentence: str, chunk_text: str, evidence_text: str) -> float:
    toks = content_tokens(sentence)
    if not toks:
        return 1.0
    support = content_tokens(chunk_text) | content_tokens(evidence_text)
    return len(toks & support) / len(toks)
