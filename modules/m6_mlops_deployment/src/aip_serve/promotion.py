"""Model registry promotion gates (promoted from NB26).

Staging is gated automatically on the NASA score (late RUL predictions cost more
than early ones). Production requires a written reason. Every event is logged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def load_ledger(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"events": [], "stages": {}}


def promote(candidate: dict, stage: str, ledger_path: str | Path,
            reason: str | None = None, actor: str = "unknown") -> dict:
    led = load_ledger(ledger_path)
    incumbent = led["stages"].get(stage)
    decision, why = "promoted", "no incumbent"
    if stage == "Production" and not reason:
        decision, why = "refused", "Production promotion requires a written reason"
    elif incumbent:
        if candidate.get("nasa_score") is None:
            decision, why = "refused", "candidate missing nasa_score"
        elif candidate["nasa_score"] > incumbent["nasa_score"]:
            decision = "refused"
            why = (f"NASA score regression: {candidate['nasa_score']} > "
                   f"incumbent {incumbent['nasa_score']}")
        else:
            why = f"NASA improved {incumbent['nasa_score']} -> {candidate['nasa_score']}"
    event = {"ts": datetime.now(UTC).isoformat(), "actor": actor, "stage": stage,
             "candidate": candidate, "decision": decision, "reason": reason, "why": why}
    led["events"].append(event)
    if decision == "promoted":
        led["stages"][stage] = candidate
    Path(ledger_path).write_text(json.dumps(led, indent=1))
    return event
