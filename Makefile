.PHONY: setup test lint up down data smoke

setup:            ## Create venv and install workspace (uv required)
	uv sync

test:             ## Run all tests
	uv run pytest

lint:             ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

up:               ## Start Phase-0 services (MLflow + Postgres)
	docker compose -f deploy/docker/docker-compose.yml up -d

up-rag:           ## Also start Qdrant (Phase 2+)
	docker compose -f deploy/docker/docker-compose.yml --profile rag up -d

down:             ## Stop all services
	docker compose -f deploy/docker/docker-compose.yml down

data:             ## Download Phase-1 datasets
	uv run python -m aip_data.download all
	uv run python -m aip_data.download all

smoke:            ## End-to-end environment smoke test
	uv run python scripts/smoke_test.py

pipeline:         ## Phase 1 end-to-end on synthetic data
	uv run python scripts/run_phase1_pipeline.py --dataset synthetic

pipeline-real:    ## Phase 1 end-to-end on real C-MAPSS FD001
	uv run python scripts/run_phase1_pipeline.py --dataset real --fd FD001

lab:              ## Launch JupyterLab
	uv run jupyter lab

nb-sync:          ## Sync all paired notebooks (.py <-> .ipynb)
	uv run jupytext --sync notebooks/**/*.py

nb-exec:          ## Execute a phase notebook chain headlessly: make nb-exec PHASE=phase2
nb-exec: PHASE ?= phase1
	cd notebooks/$(PHASE) && for f in nb*.py; do \
	  uv run jupytext --to ipynb --set-kernel python3 --execute $$f || exit 1; done

up-full:          ## Start the full stack (all services + API)
	docker compose -f deploy/docker/docker-compose.full.yml up -d

demo:             ## Run the end-to-end loop demo (NB24)
	cd notebooks/phase4 && uv run jupytext --to ipynb --set-kernel python3 --execute nb24_end_to_end_demo.py
