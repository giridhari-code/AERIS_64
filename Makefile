.PHONY: install install-dev test lint format build docker-build docker-run serve serve-chat count-1b ablation clean

PYTHON ?= python
IMAGE  = neurofield:2.3.0

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

# Requires NEUROFIELD_CHECKPOINT=path/to/dir with model.safetensors
serve:
	PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

serve-chat:
	@test -n "$$NEUROFIELD_CHECKPOINT" || (echo "Set NEUROFIELD_CHECKPOINT to a folder with model.safetensors"; exit 1)
	PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

count-1b:
	PYTHONPATH=src $(PYTHON) scripts/research/count_params.py configs/aeris_1b.yaml

ablation:
	PYTHONPATH=src $(PYTHON) scripts/research/contribution_ablation.py \
		--steps 400 --seeds 0 --device cpu --out artifacts/contrib.json

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
