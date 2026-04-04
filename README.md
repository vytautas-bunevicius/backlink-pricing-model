# Backlink Pricing Model

Machine learning pipeline for predicting backlink placement valuations based on SEO metrics from Ahrefs and Majestic.

---

## Overview

This model standardizes pricing by analyzing domain quality signals:
- **Authority**: DR (Ahrefs), CF & TF (Majestic).
- **Traffic**: Estimated organic search traffic.
- **Context**: TLD, Country, and Niche.
- **Temporal**: Historical pricing and acquisition dates.

---

## Quick Start

### 1. Requirements
- Python 3.12 or 3.13
- [uv](https://astral.sh/uv) package manager

### 2. Setup
```bash
git clone https://github.com/vytautas-bunevicius/backlink-pricing-model.git
cd backlink-pricing-model
uv sync --all-extras --dev
source .venv/bin/activate
```

---

## How It Works (Core Systems)

The codebase is organized by intent within [src/backlink_pricing_model/](src/backlink_pricing_model/).

### 1. Data Preparation ([src/backlink_pricing_model/preprocessing/](src/backlink_pricing_model/preprocessing/))
Handles the transition from raw SEO data to model-ready features:
- **Cleaning**: Outlier removal and consistency checks in [data_cleaning.py](src/backlink_pricing_model/preprocessing/data_cleaning.py).
- **Engineering**: Domain-specific feature generation (e.g., traffic-to-authority ratios) in [feature_engineering.py](src/backlink_pricing_model/preprocessing/feature_engineering.py).

### 2. Analysis & Selection ([src/backlink_pricing_model/analysis/](src/backlink_pricing_model/analysis/))
- **Selection**: Automated feature selection using correlation analysis and importance ranking in [feature_selection.py](src/backlink_pricing_model/analysis/feature_selection.py).
- **Explainability**: SHAP value calculation for model transparency in [shap_analysis.py](src/backlink_pricing_model/analysis/shap_analysis.py).

### 3. Modeling ([src/backlink_pricing_model/modeling/](src/backlink_pricing_model/modeling/))
- **Training**: Managed training loops for XGBoost and LightGBM with Optuna hyperparameter optimization.
- **AutoML**: High-accuracy ensemble modeling via [AutoGluon](https://auto.gluon.ai/).

---

## Reproducible Pipeline

A [Makefile](Makefile) is provided to orchestrate the pipeline:

| Command | Description |
| :--- | :--- |
| `make pipeline` | Full Pipeline: Extract → Preprocess → Train (AutoGluon) |
| `make extract` | Pull raw data from Supabase to `data/raw/` |
| `make preprocess` | Clean data and engineer features to `data/processed/` |
| `make train` | Train XGBoost with Optuna HPO |
| `make train-ag` | Train with AutoGluon (Best Quality mode) |
| `make evaluate` | Generate evaluation plots and metrics |

---

## Standards & Security

### Naming Conventions
- Use **snake_case** for all code and artifacts.
- Follow the Google Python Style Guide for docstrings.

### Security
- **Supabase**: Access credentials must be stored in a `.env` file (never committed) or GCP Secret Manager for production.

---

## Deployment & Maintenance

### Deployment
- **CI/CD**: Pushing to `main` triggers [GitHub Actions](.github/workflows/) for linting and testing.
- **Inference**: Use `make predict INPUT=path/to/csv` for batch predictions.

---

## Maintainers
- **Growth Marketing Tools Team**
- **Primary Contact**: Vytautas Bunevicius (vytautas.bunevicius@nordsec.com)
