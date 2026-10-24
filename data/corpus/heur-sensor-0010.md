---
doc_id: heur-sensor-0010
title: Accelerometer mounting and validity checks
source_type: authored
tags: [sensors, vibration, data-quality]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Suspicious spectra: flat noise, missing high frequencies, or ski-slope low-frequency rise.
## Heuristic
Mounting sets the usable bandwidth: stud ~10 kHz, adhesive ~5 kHz, magnet ~2 kHz, handheld ~1 kHz. Ski-slope shape usually means cable or integration artifacts, not machine fault. Validate any new alarm by checking bias voltage and repeating the measurement.
## Action window
Data-quality faults: fix before any diagnostic conclusion is recorded.
## Pitfalls
Temperature transients on magnets cause drift for the first minutes - let readings settle.
