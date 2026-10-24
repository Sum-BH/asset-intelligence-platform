# %% [markdown]
# # NB14 — Multi-Category Scale-out + Threshold Calibration (R20)
# 3 surface categories; per-category F1-optimal thresholds persisted for downstream
# use (NB18 findings, Phase 4 tools). Gate: mean image-AUROC ≥ 0.95.

# %%
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

DATA = Path("../../data")
nb13 = __import__("importlib.util", fromlist=["util"])
import importlib.util
spec = importlib.util.spec_from_file_location("nb13", "nb13_anomaly_patchcore.py")
# re-declare instead of executing NB13 (keeps runs independent):
exec(Path("nb13_anomaly_patchcore.py").read_text().split("# %% [markdown]\n# ## 3.")[0]
     .split("model = PatchCoreLite()")[0])  # defs only: generators + PatchCoreLite

# %%
report, aurocs = {}, []
for cat in range(3):
    train, test, y = make_anomaly_set(category_seed=cat)
    m = PatchCoreLite().fit(train)
    s = np.array([m.image_score(im) for im in test])
    auroc = roc_auc_score(y, s)
    # F1-optimal threshold sweep
    cands = np.quantile(s, np.linspace(0.05, 0.95, 37))
    f1s = [f1_score(y, s >= t) for t in cands]
    t_star = float(cands[int(np.argmax(f1s))])
    report[f"category{cat}"] = {"auroc": round(float(auroc), 4),
                                "threshold": round(t_star, 5),
                                "f1_at_threshold": round(max(f1s), 3)}
    aurocs.append(auroc)
    print(f"cat{cat}: AUROC={auroc:.3f}  thr={t_star:.4f}  F1={max(f1s):.3f}")

mean_auroc = float(np.mean(aurocs))
assert mean_auroc >= 0.95, "mean AUROC below gate"
json.dump({"mean_auroc": round(mean_auroc, 4), "per_category": report},
          open(DATA / "processed/calibration_report.json", "w"), indent=2)
print(f"mean image-AUROC: {mean_auroc:.3f}")
print("NB14 complete.")
