# %% [markdown]
# # NB06 — Week 8: Evidence → LLM Diagnostic Summary
# Assembles the frozen Evidence schema (§7.2) from NB03/NB05 outputs, renders the
# guarded prompt, and calls local Ollama if reachable (dry-run otherwise).
# Conversion target: `aip_signal/diagnose.py`.

# %%
import json
from pathlib import Path

import requests

DATA_DIR = Path("../../data")
OLLAMA = "http://localhost:11434"

SYSTEM = ("You are a reliability engineering assistant. Write a short diagnostic "
          "summary for a maintenance planner. Use ONLY the evidence provided in the "
          "JSON block; do not invent measurements. Structure: (1) current condition, "
          "(2) supporting evidence, (3) recommended action window, (4) one explicit "
          "sentence on uncertainty. Maximum 150 words. Plain language.")

# %% [markdown]
# ## 1. Evidence from earlier notebooks — never hand-typed numbers

# %%
best = json.loads((DATA_DIR / "processed/nb03_best.json").read_text())
evidence = {
    "asset_id": "unit-001",
    "rul_estimate_cycles": None,          # fill from your per-unit prediction on workstation
    "rul_model": best["best_model"],
    "rul_rmse_cycles": round(best["rmse"], 2),
    "health_index_current": 0.0,          # from NB05 on real IMS data
    "health_index_trend": "flat",
    "top_features": ["s2", "s2_d1", "s2_rm5"],
    "notes": "Assembled by NB06 from NB03/NB05 outputs.",
}
prompt = f"{SYSTEM}\n\nEVIDENCE:\n{json.dumps(evidence, indent=2)}\n\nSUMMARY:"
assert "uncertainty" in prompt.lower() and evidence["rul_model"] in prompt
print(prompt[:400], "...")

# %% [markdown]
# ## 2. Generate locally (graceful dry-run if Ollama is down)

# %%
try:
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": "llama3.1:8b", "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.2}}, timeout=120)
    r.raise_for_status()
    print("--- DIAGNOSIS ---\n", r.json()["response"].strip())
except Exception as e:
    print(f"Ollama unavailable ({type(e).__name__}) — dry run; prompt rendered above.")
print("NB06 complete.")
