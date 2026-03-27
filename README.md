# backlink-pricing-model

Backlink pricing prediction model using Ahrefs and Majestic SEO metrics.

Predicts the fair market price for a backlink placement based on domain quality signals:

- **DR** (Domain Rating) — Ahrefs authority score (0-100)
- **CF** (Citation Flow) — Majestic link equity score (0-100)
- **TF** (Trust Flow) — Majestic link trust score (0-100)
- **Domain traffic** — Estimated organic search traffic
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
├── notebooks/                    # Numbered Jupyter notebooks (EDA -> Modeling)
├── scripts/                      # CLI data pipelines (Supabase extraction)
├── configs/                      # YAML experiment configs
├── models/                       # Trained model artifacts (gitignored)
├── data/                         # Raw, processed, engineered data (gitignored)
├── images/                       # Saved plot outputs
├── tests/                        # Mirrors src/ structure
└── pyproject.toml                # uv + hatch + ruff + pytest config
```

## Setup

```bash
# Clone
git clone https://github.com/vytautas-bunevicius/backlink-pricing-model.git
cd backlink-pricing-model

# Install with uv
uv sync
uv sync --extra notebook    # for Jupyter
uv sync --extra extraction  # for Supabase data pipeline
uv sync --extra dev         # for development tools

# Install pre-push hook
cp .github/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Usage

### Notebooks

Run notebooks in order for the full pipeline:

1. `01_data_loading_and_eda.ipynb` — Load data, explore distributions
2. `02_feature_engineering_and_selection.ipynb` — Engineer and select features
3. `03_modeling_and_evaluation.ipynb` — Train models, evaluate, analyze with SHAP

### Data extraction

```bash
cp .env.example .env
# Fill in your Supabase credentials
python -m scripts.data_pipeline.main
```

## Development

```bash
# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

## License

[Unlicense](LICENSE) — public domain.
