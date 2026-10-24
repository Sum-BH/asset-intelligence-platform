# NUMBERS.md — measured platform figures

Every business-case number must trace to a line here. Figures marked *synthetic*
are sandbox measurements proving the code path; re-measure on real data before
publishing externally.

| Metric | Value | Source artifact |
|---|---|---|
| RUL best baseline | gradient_boosting RMSE 8.73 cycles | `nb03_best.json (synthetic FD00S; re-measure on real FD001)` |
| Visual anomaly detection | mean image-AUROC 1.0 | `calibration_report.json` |
| Defect detection | mAP@0.5 0.6315 (classical baseline) | `nb15_detections_meta.json` |
| Thermal hotspots | recall@IoU0.5 0.9048 | `nb16_hotspots.json` |
| FOD detection | mAP@0.5 1.0 | `nb17_fod.json` |
| Agent eval | 11 scenarios, groundedness 1.0, 5 declines | `nb23_eval_baseline.json` |
| Scheduler scale | 50 WOs placed in 10.01s (FEASIBLE, optimality proven: False) | `nb21_schedule.json` |
| End-to-end demo | 3 alarms → 2 scheduled in 1.2s | `nb24_demo_audit.json` |
| Drift monitoring | latency 1 windows, stationary FP rate 0.0 | `nb27_drift.json` |
