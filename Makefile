.PHONY: install install-dev test lint format build docker-build docker-run backend frontend serve serve-chat clean

PYTHON ?= python
IMAGE  = sentinel:2.3.1
export PYTHONPATH := packages$(if $(PYTHONPATH),:$(PYTHONPATH),)

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,serve]"

test:
	PYTHONPATH=packages $(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check packages/sentinel tests/

format:
	ruff format packages/sentinel tests/

build:
	$(PYTHON) -m build

docker-build:
	docker build -t $(IMAGE) -f deploy/docker/Dockerfile .

docker-run:
	docker compose -f deploy/docker/docker-compose.yml up --build

backend:
	bash backend/run.sh

serve: backend

frontend:
	cd frontend && bun install && bun run dev

serve-chat:
	@echo "Terminal 1: make backend"
	@echo "Terminal 2: make frontend"
	@echo "Open http://localhost:3000"

clean:
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info 2>/dev/null || true
