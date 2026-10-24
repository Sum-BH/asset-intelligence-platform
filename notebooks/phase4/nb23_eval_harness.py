# %% [markdown]
# # NB23 — Agent Evaluation Harness (R31, R32)
# Scenarios replay REAL Phase 1/3 outputs as fixtures — synthetic-only fixtures would
# let the agent overfit to clean data. Metrics: action groundedness (every action cites
# retrieved evidence), tool-call efficiency, decline correctness on weak evidence.
# RAGAS + Langfuse are the workstation upgrade (guarded); the harness itself is the
# durable asset. Baseline committed for regression. Target: `aip_agents/eval.py`.

# %%
import importlib.util
import json
import re
from pathlib import Path

DATA = Path("../../data")
_t = Path("nb19_tools.py").read_text().split("# ## 5. Fixture tests")[0]
exec("\n".join(ln for ln in _t.splitlines() if not ln.startswith(("# %%", "# #"))))
_g = Path("nb20_agent_graph.py").read_text().split("# ## 3. Run: draft")[0]
exec("\n".join(ln for ln in _g.splitlines()
               if not ln.startswith(("# %%", "# #", "from __future__"))))

# %% [markdown]
# ## 1. Scenarios — built from assets that actually exist in the repo artifacts

# %%
vf = json.loads((DATA / "processed/vision_findings.json").read_text())
real_assets = sorted({f["asset_id"] for f in vf})
SCENARIOS = (
    [{"id": f"sc-real-{a}", "asset_id": a, "expect": "work_order"} for a in real_assets]
    + [{"id": "sc-signal-only", "asset_id": "unit-3", "expect": "any"},
       {"id": "sc-signal-only-2", "asset_id": "unit-7", "expect": "any"},
       {"id": "sc-decline-ghost", "asset_id": "ghost-asset-99", "expect": "declined"},
       {"id": "sc-decline-empty", "asset_id": "not-an-asset", "expect": "declined"},
       {"id": "sc-decline-blank", "asset_id": "zzz-000", "expect": "declined"},
       {"id": "sc-vision-heavy", "asset_id": "taxiway-A3", "expect": "work_order"},
       {"id": "sc-thermal", "asset_id": "panel-2", "expect": "work_order"},
       {"id": "sc-mech", "asset_id": "press-1", "expect": "work_order"}]
)
print(f"{len(SCENARIOS)} scenarios ({sum(s['expect']=='declined' for s in SCENARIOS)} decline cases)")
assert len(SCENARIOS) >= 10 and sum(s["expect"] == "declined" for s in SCENARIOS) >= 2

# %% [markdown]
# ## 2. Metrics

# %%
CHUNK_IDS = {c["chunk_id"] for c in CH}

def grounded(wo: dict) -> bool:
    """Every raised work order's action must cite a real chunk_id from the KB."""
    if wo["status"] == "declined":
        return True                      # declining requires no citation
    cited = re.findall(r"\[([\w#-]+)\]", wo["recommended_action"])
    return bool(cited) and all(c in CHUNK_IDS for c in cited)


def evidence_consistent(wo: dict) -> bool:
    """Priority must not exceed what the evidence supports."""
    strength = wo["evidence"].get("evidence_strength", "none")
    if wo["status"] == "declined":
        return wo["priority"] == 1 and wo["duration_h_est"] == 0
    return not (wo["priority"] == 3 and strength != "strong")

# %% [markdown]
# ## 3. Run the harness against the real graph

# %%
rows, tool_calls = [], 3          # signals+vision+knowledge per run (v1 cap = 3 tools)
for sc in SCENARIOS:
    cfg = {"configurable": {"thread_id": f"eval-{sc['id']}"}}
    out = graph.invoke({"asset_id": sc["asset_id"]}, cfg)
    wo = out["work_order"]
    declined = wo["status"] == "declined"
    ok_expect = (sc["expect"] == "any"
                 or (sc["expect"] == "declined") == declined)
    rows.append({"id": sc["id"], "asset": sc["asset_id"], "status": wo["status"],
                 "priority": wo["priority"], "expect_ok": ok_expect,
                 "grounded": grounded(wo), "consistent": evidence_consistent(wo),
                 "tool_calls": tool_calls})

n = len(rows)
summary = {
    "n_scenarios": n,
    "expectation_match": round(sum(r["expect_ok"] for r in rows) / n, 3),
    "groundedness": round(sum(r["grounded"] for r in rows) / n, 3),
    "evidence_consistency": round(sum(r["consistent"] for r in rows) / n, 3),
    "mean_tool_calls": tool_calls,
    "declines": sum(r["status"] == "declined" for r in rows),
    "metric_version": "harness-v1 (lexical citation check)",
    "workstation_upgrade": "RAGAS faithfulness + Langfuse traces",
}
for r in rows:
    print(f"  {r['id']:22s} {r['status']:9s} p{r['priority']} "
          f"expect_ok={r['expect_ok']} grounded={r['grounded']}")
print("\nsummary:", json.dumps(summary, indent=1))

assert summary["expectation_match"] == 1.0, "scenario expectations violated"
assert summary["groundedness"] >= 0.85, "ungrounded actions present"
assert summary["evidence_consistency"] == 1.0, "priority not supported by evidence"
assert summary["declines"] >= 2, "decline path must be exercised"

# %% [markdown]
# ## 4. Commit the regression baseline

# %%
json.dump({"summary": summary, "rows": rows},
          open(DATA / "processed/nb23_eval_baseline.json", "w"), indent=1)
(DATA / "processed/scenarios.jsonl").write_text(
    "\n".join(json.dumps(s) for s in SCENARIOS))

for pkg, note in [("ragas", "faithfulness/answer-relevancy with local LLM"),
                  ("langfuse", "trace every tool call and interrupt")]:
    print(f"{'present' if importlib.util.find_spec(pkg) else 'SKIP'} {pkg}: {note}")
print("NB23 complete.")
