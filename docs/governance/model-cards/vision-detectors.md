# Model Card — Vision Detectors (anomaly, defect, thermal, FOD)

**Version:** Phase 3 · **Owner:** Sumedh Bhalerao · **Status:** Staging

## Intended use
Flag candidate surface defects, thermal hotspots and foreign objects for human review.
The detectors decide pass/fail; the vision-language model only describes what is visible.

## Data
Synthetic imagery from the repository generators (sandbox) and public research datasets
(MVTec AD, NEU, FOD-A) on workstation. **MVTec AD is non-commercial** — portfolio use
only; commercial engagements must use client data.

## Metrics
Image-AUROC (anomaly), mAP@0.5 (defect, FOD), recall@IoU0.5 (thermal). Values in
`docs/business/NUMBERS.md`. Sandbox figures come from synthetic imagery whose difficulty
is controlled by the generator — they validate the pipeline, not real-world accuracy.

## Limitations
- Classical sandbox detectors are baselines to be beaten by fine-tuned YOLO on real data.
- Thermal detection assumes background-relative ΔT; emissivity and reflections mislead —
  minimum load conditions apply for electrical inspection.
- The VLM must not speculate on cause; prompts constrain it to visible content only.

## Drift triggers
Distribution shift in detection counts per shift, or a rise in reviewer rejections.
