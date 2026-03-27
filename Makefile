.PHONY: setup extract preprocess train evaluate predict test lint format clean help

PYTHON := python
CONFIG_PREPROCESS := configs/preprocessing.yaml
CONFIG_TRAIN := configs/training.yaml

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

setup: ## Install all dependencies (core + dev + notebook + extraction)
	uv sync --extra dev --extra notebook --extra extraction
	cp -n .github/hooks/pre-push .git/hooks/pre-push 2>/dev/null || true
	chmod +x .git/hooks/pre-push 2>/dev/null || true
	@echo "Setup complete."

# ── Data Pipeline ────────────────────────────────────────────────────

extract: ## Extract raw data from Supabase to data/raw/
	$(PYTHON) -m scripts.data_pipeline.main

preprocess: ## Clean and feature-engineer raw data
	$(PYTHON) -m scripts.preprocess --config $(CONFIG_PREPROCESS)

# ── Training ─────────────────────────────────────────────────────────

train: ## Train model with Optuna HPO (use CONFIG= to override)
	$(PYTHON) -m scripts.train --config $(CONFIG_TRAIN)

train-quick: ## Quick training run with 10 Optuna trials
	$(PYTHON) -m scripts.train --config $(CONFIG_TRAIN) --trials 10

# ── Evaluation & Inference ───────────────────────────────────────────

evaluate: ## Evaluate trained model on test set
	$(PYTHON) -m scripts.evaluate --config $(CONFIG_TRAIN)

predict: ## Run inference (use INPUT= to specify file)
	@test -n "$(INPUT)" || (echo "Usage: make predict INPUT=data/raw/new_domains.csv" && exit 1)
	$(PYTHON) -m scripts.predict --config $(CONFIG_TRAIN) --input $(INPUT)

# ── Full Pipeline ────────────────────────────────────────────────────

pipeline: extract preprocess train evaluate ## Run full pipeline: extract → preprocess → train → evaluate

retrain: preprocess train evaluate ## Retrain from existing raw data (skip extraction)

# ── Quality ──────────────────────────────────────────────────────────

test: ## Run all tests
	pytest tests/

lint: ## Run linter checks
	ruff check .

format: ## Auto-format code
	ruff format .

check: lint test ## Run lint + tests

# ── Cleanup ──────────────────────────────────────────────────────────

clean: ## Remove generated artifacts (data, models, images)
	rm -rf data/processed/*.parquet data/engineered/*.parquet
	rm -rf models/*.joblib models/*.json models/*.csv models/*.db
	rm -rf images/modeling/*.png images/eda/*.png
	@echo "Cleaned generated artifacts."

clean-all: clean ## Also remove raw data (requires re-extraction)
	rm -rf data/raw/*.parquet data/raw/*.csv
	@echo "Cleaned all data including raw."
