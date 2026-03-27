.PHONY: setup extract preprocess train train-quick train-autogluon evaluate predict pipeline retrain test lint format check clean clean-all help

PYTHON := python
CONFIG_PREPROCESS := configs/preprocessing.yaml
CONFIG_TRAIN := configs/training.yaml

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies (core + dev + notebook + extraction + autogluon)
	uv sync --extra dev --extra notebook --extra extraction --extra autogluon
	cp -n .github/hooks/pre-push .git/hooks/pre-push 2>/dev/null || true
	chmod +x .git/hooks/pre-push 2>/dev/null || true
	@echo "Setup complete."

extract: ## Extract raw data from Supabase to data/raw/
	$(PYTHON) -m scripts.data_pipeline.main

preprocess: ## Clean and feature-engineer raw data
	$(PYTHON) -m scripts.preprocess --config $(CONFIG_PREPROCESS)

train: ## Train XGBoost with Optuna HPO
	$(PYTHON) -m scripts.train --config $(CONFIG_TRAIN)

train-quick: ## Quick XGBoost training (10 Optuna trials)
	$(PYTHON) -m scripts.train --config $(CONFIG_TRAIN) --trials 10

train-autogluon: ## Train with AutoGluon (best_quality, 1h default)
	$(PYTHON) -m scripts.train_autogluon --config $(CONFIG_TRAIN)

train-autogluon-quick: ## Quick AutoGluon training (10 min)
	$(PYTHON) -m scripts.train_autogluon --config $(CONFIG_TRAIN) --time-limit 600

evaluate: ## Evaluate trained model on test set
	$(PYTHON) -m scripts.evaluate --config $(CONFIG_TRAIN)

predict: ## Run inference (use INPUT= to specify file)
	@test -n "$(INPUT)" || (echo "Usage: make predict INPUT=data/raw/new_domains.csv" && exit 1)
	$(PYTHON) -m scripts.predict --config $(CONFIG_TRAIN) --input $(INPUT)

pipeline: extract preprocess train-autogluon ## Full pipeline with AutoGluon

retrain: preprocess train-autogluon ## Retrain from existing raw data

test: ## Run all tests
	pytest tests/

lint: ## Run linter checks
	ruff check .

format: ## Auto-format code
	ruff format .

check: lint test ## Run lint + tests

clean: ## Remove generated artifacts (data, models, images)
	rm -rf data/processed/*.parquet data/engineered/*.parquet
	rm -rf models/*.joblib models/*.json models/*.csv models/*.db
	rm -rf models/autogluon/
	rm -rf images/modeling/*.png images/eda/*.png
	@echo "Cleaned generated artifacts."

clean-all: clean ## Also remove raw data (requires re-extraction)
	rm -rf data/raw/*.parquet data/raw/*.csv
	@echo "Cleaned all data including raw."
