# ADR-002: Notebook-first development with jupytext pairing; terminal conversion phase

- **Status:** Accepted
- **Date:** 2026-07-15
- **Deciders:** Sumedh Bhalerao

## Context
The owner works fastest in Jupyter notebooks and wants all phase work done
notebook-first, with conversion to production files at the end of the project.
Pure-ipynb workflows have known failure modes: unreviewable git diffs, merge
conflicts, silent code drift, and a high-risk big-bang conversion after months
of accumulation.

## Options considered
1. **Module-first (previous plan).** Cleanest engineering; fights the owner's
   actual working style, so it would silently not be followed. Rejected.
2. **Pure .ipynb in git, big-bang conversion in the final phase.** Matches the
   stated wish literally, but 30+ notebooks of untested drifted code converted
   at once is where notebook projects classically die. Rejected.
3. **Notebook-first with jupytext pairing (chosen).** Author exclusively in
   notebooks; jupytext auto-maintains a py:percent twin (the reviewable git
   source). Notebooks must still WRITE the frozen interface artifacts (Phase 1
   SOP section 7.2), so later phases are unaffected by where code lives. The
   final conversion phase (Phase 6) promotes stable notebook code into the
   module skeleton — mechanical, because the py twins are already clean Python
   and the verified reference modules in the repo serve as the conversion
   target/answer key.

## Decision
Adopt option 3. Conventions: notebooks live in notebooks/phase<N>/ named
nbNN_topic; jupytext.toml pairs ipynb <-> py:percent; `make nb-exec` must run
the phase's notebook chain end-to-end with zero errors before phase exit
(notebooks self-skip hardware-gated cells, e.g. torch). Interface contracts
remain the acceptance criteria regardless of authoring medium.

## Consequences
+ Owner works in the medium he is fastest in; git history stays reviewable.
+ Phase 6 conversion is a short mechanical promotion, not a rewrite.
+ Executable-notebook rule preserves "verified and working" discipline.
- Two artifacts per notebook (.py/.ipynb) must stay in sync (`make nb-sync`).
- Notebook code duplicates reference-module logic until Phase 6 merges them.
