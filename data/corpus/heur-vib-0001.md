---
doc_id: heur-vib-0001
title: Bearing outer-race defects in envelope spectra
source_type: authored
tags: [bearings, vibration, envelope]
author: Sumedh Bhalerao
date: 2026-07-15
review_status: DRAFT - starter content generated for structure; REWRITE in your own words and experience before publishing
---
## Symptom
Distinct peaks at BPFO (ball-pass frequency, outer race) and its harmonics in the envelope spectrum, often before any change in overall RMS.
## Heuristic
Compute the envelope spectrum from a band around the bearing resonance (typically 2-6 kHz for medium machines). If BPFO amplitude exceeds 3x its healthy baseline for two consecutive measurements, flag stage-2 bearing damage.
## Action window
Plan replacement within 4-8 weeks of stage-2 confirmation for pumps and fans; sooner under shock loading.
## Pitfalls
Slip makes real frequencies ~1-2% below theoretical. A single high reading after maintenance is often a mounting artifact - confirm on the next sample.
