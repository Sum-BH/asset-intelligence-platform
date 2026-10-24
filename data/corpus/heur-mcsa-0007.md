---
doc_id: heur-mcsa-0007
title: Motor current signature basics for mechanical faults
source_type: authored
tags: [motors, mcsa, electrical]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Sidebands around line frequency in stator current spectrum.
## Heuristic
Pole-pass sidebands (slip x poles) above -45 dB relative to line frequency indicate rotor-bar issues. Load-oscillation sidebands can also reveal downstream mechanical faults (belts, couplings) without a vibration sensor.
## Action window
Rotor-bar indication: confirm with a second measurement at stable load, then plan repair at next major stop.
## Pitfalls
VFDs pollute the spectrum; measure on the line side or use the drive's own diagnostics.
