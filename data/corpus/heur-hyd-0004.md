---
doc_id: heur-hyd-0004
title: Hydraulic system contamination heuristics
source_type: authored
tags: [hydraulics, contamination, filtration]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Rising differential pressure across filters, erratic valve response, elevated particle counts (ISO 4406 codes climbing).
## Heuristic
A two-code jump in any ISO 4406 channel between samples indicates an active wear or ingression event, not measurement noise. Correlate with recent maintenance openings.
## Action window
Two-code jump: resample within one week and inspect breathers/seals. Sustained: flush and root-cause before component failure cascades.
## Pitfalls
Sampling technique dominates variance - always sample from the same live point at operating temperature.
