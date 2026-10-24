"""LLM diagnostic summary: structured evidence -> plain-language diagnosis.

Evidence is assembled from pipeline outputs (never free text from the model's
imagination): RUL estimate, health-index trend, top contributing features.
The prompt forces the model to (a) only reference provided evidence and
(b) state uncertainty explicitly.

dry_run=True returns the rendered prompt without calling Ollama — used in
tests and environments without a GPU/Ollama.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import requests
from aip_core import settings
from aip_core.logging import get_logger

log = get_logger("aip_signal.diagnose")

SYSTEM = (
    "You are a reliability engineering assistant. Write a short diagnostic "
    "summary for a maintenance planner. Use ONLY the evidence provided in the "
    "JSON block; do not invent measurements. Structure: (1) current condition, "
    "(2) supporting evidence, (3) recommended action window, (4) one explicit "
    "sentence on uncertainty. Maximum 150 words. Plain language."
)


@dataclass
class Evidence:
    asset_id: str
    rul_estimate_cycles: float
    rul_model: str
    rul_rmse_cycles: float
    health_index_current: float
    health_index_trend: str  # "flat" | "rising" | "falling"
    top_features: list[str] = field(default_factory=list)
    notes: str = ""


def render_prompt(ev: Evidence) -> str:
    return f"{SYSTEM}\n\nEVIDENCE:\n{json.dumps(asdict(ev), indent=2)}\n\nSUMMARY:"


def diagnose(
    ev: Evidence, model: str = "llama3.1:8b", dry_run: bool = False, timeout: int = 120
) -> str:
    prompt = render_prompt(ev)
    if dry_run:
        return prompt
    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    log.info("diagnosis generated (%d chars) with %s", len(text), model)
    return text
