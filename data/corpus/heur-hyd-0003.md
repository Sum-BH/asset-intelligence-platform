---
doc_id: heur-hyd-0003
title: Pump cavitation signatures
source_type: authored
tags: [hydraulics, pumps, cavitation]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Broadband noise floor rise in 1-20 kHz, random impacting in the time waveform, crackling audible at the suction side; discharge pressure fluctuation.
## Heuristic
If broadband energy rises >6 dB with no discrete peak growth, suspect cavitation before mechanical damage. Verify NPSH margin: available minus required should exceed 1 m; below that, cavitation is near-certain at duty flow.
## Action window
Operational fix first (raise suction level, trim flow); persistent cavitation erodes impellers in weeks to months.
## Pitfalls
Recirculation at low flow mimics cavitation; check where the duty point sits on the curve.
