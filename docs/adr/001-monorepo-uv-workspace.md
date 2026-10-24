# ADR-001: Monorepo with uv workspaces; DuckDB-first local storage

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Sumedh Bhalerao

## Context
The platform has six modules that must be independently demoable and sellable,
yet share config, logging, and a dataset registry. Time budget is 5-8 h/week,
so tooling friction must be near zero. Hardware is a single RTX 4060 Ti
workstation; everything must run locally with Databricks only as an optional
scale-out story.

## Options considered
1. **Multi-repo (one repo per module).** Clean isolation, but shared-code drift,
   6x CI/setup overhead, and a fragmented portfolio story. Rejected.
2. **Monorepo, single flat package.** Simple, but modules can't be versioned,
   packaged, or demoed independently; freelance one-pagers need per-module
   artifacts. Rejected.
3. **Monorepo with uv workspaces (chosen).** One lockfile, one `uv sync`, each
   module is a real installable package (`aip-data`, `aip-signal`, ...); the
   root project pins only the members active in the current phase.

For storage: Postgres-first was considered, but DuckDB + Parquet gives
zero-ops local analytics, direct Parquet scanning, and an easy Delta/Databricks
migration path. Postgres is retained solely as the MLflow backend store.

## Decision
Adopt a uv-workspace monorepo (`shared/aip_core` + `modules/m1..m6`), hatchling
builds, ruff + pytest at root, Docker Compose with profiles so later-phase
services (Qdrant, Redpanda) stay dormant until their phase starts.

## Consequences
+ One-command setup; per-module packaging preserved for freelance demos.
+ Phase discipline enforced structurally: root deps only include active modules.
- Workspace requires uv (not plain pip) — acceptable, documented in README.
- Compose profiles must be kept in sync with the roadmap.
