import json

from aip_signal.diagnose import Evidence, diagnose, render_prompt


def test_prompt_contains_all_evidence():
    ev = Evidence(
        asset_id="A-1",
        rul_estimate_cycles=42.0,
        rul_model="rf",
        rul_rmse_cycles=12.3,
        health_index_current=0.8,
        health_index_trend="rising",
        top_features=["s2_rm10"],
    )
    prompt = render_prompt(ev)
    payload = json.loads(prompt.split("EVIDENCE:\n")[1].rsplit("\n\nSUMMARY:", 1)[0])
    assert payload["asset_id"] == "A-1"
    assert payload["health_index_trend"] == "rising"
    assert "uncertainty" in prompt.lower()


def test_dry_run_no_network():
    ev = Evidence(
        asset_id="A-2",
        rul_estimate_cycles=10,
        rul_model="gb",
        rul_rmse_cycles=5,
        health_index_current=0.1,
        health_index_trend="flat",
    )
    out = diagnose(ev, dry_run=True)
    assert out.startswith("You are a reliability")
