---
doc_id: heur-thermal-0008
title: Thermal image interpretation for electrical panels
source_type: authored
tags: [thermal, electrical, inspection]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Localized hotspots on connections, phase imbalance visible as one conductor warmer than siblings.
## Heuristic
Judge by delta-T between like components under similar load, not absolute temperature. Delta-T 5-15 degC: plan repair; 15-40 degC: repair soon; >40 degC: immediate.
## Action window
Per delta-T bands above; always re-scan after torque correction.
## Pitfalls
Emissivity and reflections lie - shiny lugs read cold. Load must exceed ~40% for meaningful comparison.
