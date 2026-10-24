# Business Case — Asset Intelligence Platform

*Draft v1. All platform figures trace to [`NUMBERS.md`](NUMBERS.md). Cost ranges are
illustrative parameters, not researched market data — **replace with sourced figures
before any external use**.*

## 1. The problem is the decision loop, not the data

Most plants already have sensors, inspection photos, thermal scans and decades of
maintenance knowledge. What they lack is a path from signal to *approved, scheduled
action*. Existing PdM tools stop at the alarm: they detect, then hand a human a
dashboard and a blank work-order form. The value leak is between detection and action.

**Thesis:** detection is commoditising; loop closure is where value migrates.

## 2. What this platform does differently

| Dimension | Typical PdM tool | This platform |
|---|---|---|
| Scope | Detect → alarm | sense → detect → diagnose → explain → prescribe → schedule → act → learn |
| Modality | Single (usually vibration) | Signals + vision + thermal + documents |
| Deployment | Cloud-locked | On-premise, inside the plant's four walls |
| Explanation | Score, sometimes a plot | Cited report tracing every claim to a knowledge-base chunk |
| Action | Human retypes into CMMS | Evidenced work-order draft → human approval → constraint-optimal schedule |
| Hardware | Enterprise infrastructure | One desktop GPU workstation |

## 3. Where the money is (parameterise per client)

Inputs to size any engagement:

- `D` = downtime cost per hour for the asset class (varies by orders of magnitude:
  a packaging line and a semiconductor tool are not comparable — **get this number
  from the client's finance team, not from a vendor whitepaper**)
- `H` = unplanned downtime hours per year on the covered assets
- `r` = realistic reduction fraction from earlier detection plus better scheduling
- `L` = planner/engineer hours per month spent assembling evidence manually

**Annual benefit ≈ (D × H × r) + (L × 12 × loaded hourly rate)**

The second term is the one clients underestimate and the one this platform attacks
most directly: the diagnostic report and work-order draft are generated from evidence
in seconds (measured: 3 alarms → 2 scheduled work orders in 1.2 s end-to-end).

Be conservative on `r`. Published PdM claims are typically vendor-sourced; a defensible
pilot posture is to *measure* `r` on a small asset set rather than promise it.

## 4. Cost side

- Hardware: one workstation-class GPU box (the reference build runs on an RTX 4060 Ti
  16 GB / 64 GB RAM machine)
- Software: open source throughout; no per-seat or per-asset licensing
- Integration: the dominant real cost — historian/CMMS connectors, sensor access,
  corpus authoring with the client's own SMEs

## 5. Why on-premise matters commercially

Process data, defect imagery and maintenance know-how are competitive assets. A
cloud-locked PdM tool asks the client to export all three. This platform runs local
LLMs, a local vector store and a local database — nothing leaves the plant. For
defence-adjacent, aviation and pharma clients this is frequently the deciding factor,
not a nice-to-have.

## 6. Pilot proposal shape (what to actually sell)

1. **Weeks 1–2 — Instrument.** Ingest one asset family's historian data; stand up KPIs.
2. **Weeks 3–6 — Detect + diagnose.** Anomaly/RUL models on their data; author 20–30
   heuristics with their SMEs (this becomes *their* knowledge asset).
3. **Weeks 7–10 — Close the loop.** Evidenced work orders, human approval gate,
   schedule optimisation against their technician roster.
4. **Exit:** measured `r` on the pilot asset set, plus a deploy-in-a-day artifact.

## 7. Risk register (state these up front — it builds credibility)

| Risk | Honest position |
|---|---|
| Model accuracy on client data | Unknown until measured; the pilot exists to find out |
| Knowledge-base quality | Depends on SME time; budget it explicitly |
| Change management | Planners must trust the drafts; the approval gate is deliberate |
| Local LLM limits | Report generation is constrained and cited; the system refuses rather than invents when coverage is thin |

## 8. What is proven today vs claimed

Proven in the repository (see NUMBERS.md): the full loop executes end-to-end on one
machine, with human approval enforced structurally, citations on every action, and a
constraint-optimal schedule. **Not yet proven:** performance on any real client's data.
The pilot structure above exists precisely to close that gap honestly.
