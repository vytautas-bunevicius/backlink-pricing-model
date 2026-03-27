# backlink-pricing-model

Backlink pricing prediction model using Ahrefs and Majestic SEO metrics.

Predicts the fair market price for a backlink placement based on domain quality signals:

- **DR** (Domain Rating) — Ahrefs authority score (0-100)
- **CF** (Citation Flow) — Majestic link equity score (0-100)
- **TF** (Trust Flow) — Majestic link trust score (0-100)
- **Domain traffic** — Estimated organic search traffic
- **TLD** — Top-level domain (.com, .co.uk, .io, etc.)
- **Country** — Geographic market of the domain

## Project structure

```
backlink-pricing-model/
├── src/backlink_pricing_model/   # Importable Python package
│   ├── core/                     # Logging, env config, Pydantic models
│   ├── preprocessing/            # Data loading, cleaning, feature engineering
│   ├── analysis/                 # Statistical tests, feature selection, SHAP
│   ├── modeling/                 # Training, baselines, hyperparameter tuning
│   ├── visualization/            # EDA, importance, and evaluation plots
│   └── utils/                    # Shared helpers
├── scripts/                      # CLI pipeline scripts
│   ├── data_pipeline/            # Supabase data extraction
│   ├── preprocess.py             # Data cleaning and feature engineering
│   ├── train.py                  # Model training with Optuna HPO
│   ├── evaluate.py               # Model evaluation and plots
│   └── predict.py                # Inference on new domains
├── notebooks/                    # Jupyter notebooks (exploration/docs)
├── configs/                      # YAML experiment configs
├── models/                       # Trained artifacts (gitignored)
├── data/                         # Raw, processed, engineered (gitignored)
├── images/                       # Saved plot outputs
├── tests/                        # Mirrors src/ structure
├── Makefile                      # Reproducible pipeline commands
└── pyproject.toml                # uv + hatch + ruff + pytest config
```

## Setup

```bash
# Clone
git clone https://github.com/vytautas-bunevicius/backlink-pricing-model.git
cd backlink-pricing-model

# Install everything
make setup

# Or manually with uv
uv sync --extra dev --extra notebook --extra extraction
```

## Reproducible pipeline

Run the full pipeline with a single command, or each step individually:

```bash
# Full pipeline: extract -> preprocess -> train -> evaluate
make pipeline

# Or step by step:
make extract       # Pull data from Supabase -> data/raw/
make preprocess    # Clean and engineer features -> data/processed/
make train         # Train XGBoost with Optuna HPO -> models/
make evaluate      # Evaluate on test set -> images/modeling/
```

### Retrain on existing data

```bash
make retrain       # preprocess -> train -> evaluate (skips extraction)
```

### Quick experiment

```bash
make train-quick   # Only 10 Optuna trials for fast iteration
```

### Custom config

```bash
# Copy and edit a config for a new experiment
cp configs/training.yaml configs/experiment_v2.yaml
# Edit experiment_v2.yaml...

make train CONFIG_TRAIN=configs/experiment_v2.yaml
```

### Predict on new domains

```bash
make predict INPUT=data/raw/new_domains.csv
```

Input CSV needs columns: `domain`, `dr`, `cf`, `tf`, `domain_traffic`, `country`, `date_received`

### All available commands

```bash
make help
```

## Notebooks

Notebooks are for exploration and documentation, not the source of truth for training. Run them after `make preprocess` to explore the data:

1. `01_data_loading_and_eda.ipynb` — Load data, explore distributions
2. `02_feature_engineering_and_selection.ipynb` — Engineer and select features
3. `03_modeling_and_evaluation.ipynb` — Train models, evaluate, analyze with SHAP

## Experiment tracking

Optuna studies are stored in `models/optuna_studies.db` (SQLite). This lets you:

- Resume interrupted training runs
- Compare trials across experiments
- Inspect hyperparameter importance with `optuna.visualization`

Training metadata (params, metrics, timestamp) is saved to `models/training_metadata.json` after each run.

## Development

```bash
make check         # Run lint + tests
make format        # Auto-format code
make clean         # Remove generated artifacts
```

## License

[Unlicense](LICENSE) — public domain.
