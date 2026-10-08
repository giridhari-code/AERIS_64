# --- 1B ---
.PHONY: count-1b
.PHONY: install install-dev test lint format build docker-build docker-run serve serve-chat clean

PYTHON ?= python
PACKAGE = neurofield
IMAGE  = neurofield:2.2.0

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,serve]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

build:
	$(PYTHON) -m build

docker-build:
	docker build -t $(IMAGE) -f deploy/docker/Dockerfile .

docker-run:
	docker compose -f deploy/docker/docker-compose.yml up --build

# API only (Swagger at /docs)
serve:
	PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Chat frontend + API (open http://127.0.0.1:8000/)
serve-chat:
	NEUROFIELD_CHECKPOINT=$${NEUROFIELD_CHECKPOINT:-docs/neurofield_india} PYTHONPATH=src \
	uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

count-1b:
	$(PYTHON) scripts/count_params.py configs/aeris_1b.yaml

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
