# 15-Minute Quickstart

## Prerequisites
Docker, [uv](https://docs.astral.sh/uv/), and (for local LLM features) Ollama.

## 1. Install and verify (3 min)
```bash
make setup          # create venv, install the uv workspace
make test           # module tests
make smoke          # environment readiness check
```

## 2. Start services (2 min)
```bash
make up             # MLflow (:5000) + Postgres
make up-full        # everything: + Qdrant, Redpanda, API (:8000)
curl localhost:8000/health
```

## 3. Run a phase notebook chain (10 min)
```bash
make data                       # download public datasets
make nb-exec PHASE=phase1       # ingestion -> features -> models -> diagnosis
make nb-exec PHASE=phase4       # tools -> agent -> optimizer -> MCP -> demo
```

## 4. See the loop close
```bash
make demo           # NB24: alarms -> drafts -> approval -> optimal schedule
```

All datasets are public; no proprietary data is used anywhere in this project.
