# Backlink Pricing Model

The Backlink Pricing Model is a professional-grade machine learning pipeline designed to predict fair market valuations for backlink placements by analyzing a comprehensive suite of SEO metrics. This project centralizes our valuation logic, allowing us to move away from subjective estimates toward data-driven decisions based on objective signals from industry-standard tools like Ahrefs and Majestic.

---

## Overview and Valuation Logic

This model aims to standardize how we price external placements by processing several key domain quality signals into a unified prediction. The valuation process considers multiple dimensions of a domain's health, including **authority metrics** like Domain Rating (DR) and Trust Flow (TF), as well as **estimated organic traffic** which provides a proxy for real-world visibility. Beyond these core metrics, the model also accounts for contextual factors such as the Top-Level Domain (TLD) and niche relevance, while identifying temporal trends through historical acquisition dates to ensure that our pricing remains accurate as market conditions evolve.

---

## Quick Start

### 1. Environment Requirements
This project is built using Python 3.12 or 3.13. We recommend using the [uv](https://astral.sh/uv) package manager to manage the virtual environment, as it provides significantly faster dependency resolution and ensures that all contributors are working with identical package versions.

### 2. Initial Setup
To get started, clone the repository and use `uv` to synchronize the environment, which will install all necessary development tools, notebook support, and machine learning libraries like AutoGluon and XGBoost.
```bash
git clone https://github.com/vytautas-bunevicius/backlink-pricing-model.git
cd backlink-pricing-model
uv sync --all-extras --dev
source .venv/bin/activate
```

---

## How It Works (Core Systems)

The codebase is organized into modular components within the [src/backlink_pricing_model/](src/backlink_pricing_model/) package, each responsible for a specific stage of the machine learning lifecycle.

### 1. Data Preparation and Feature Engineering
The transition from raw SEO data to model-ready features is handled by our [preprocessing module](src/backlink_pricing_model/preprocessing/). This includes rigorous data cleaning in [data_cleaning.py](src/backlink_pricing_model/preprocessing/data_cleaning.py) to remove outliers and handle missing values, followed by domain-specific feature generation in [feature_engineering.py](src/backlink_pricing_model/preprocessing/feature_engineering.py) where we calculate advanced signals like traffic-to-authority ratios.

### 2. Automated Analysis and Selection
To ensure our models remain both accurate and interpretable, the [analysis module](src/backlink_pricing_model/analysis/) performs automated feature selection. We use [feature_selection.py](src/backlink_pricing_model/analysis/feature_selection.py) to rank features by their importance and remove redundant correlations, while [shap_analysis.py](src/backlink_pricing_model/analysis/shap_analysis.py) provides transparency by calculating SHAP values to explain exactly how each input influences the final price prediction.

### 3. Training and Modeling
Our [modeling surface](src/backlink_pricing_model/modeling/) supports multiple training paths depending on the required accuracy. We maintain managed training loops for gradient-boosted trees via XGBoost and LightGBM with Optuna-based hyperparameter tuning, alongside a high-accuracy ensemble path provided by [AutoGluon](https://auto.gluon.ai/), which serves as our primary model for production valuations.

---

## Reproducible Pipeline

We provide a comprehensive [Makefile](Makefile) to orchestrate the entire end-to-end pipeline, ensuring that every stage from data extraction to model evaluation can be reproduced with a single command. You can use `make pipeline` to run the full flow, or execute individual stages like `make extract` to pull fresh data from Supabase or `make evaluate` to generate performance metrics and visualization plots for a recently trained model.

---

## Standards and Security

To maintain a consistent development experience, we follow the Google Python Style Guide for all docstrings and enforce **snake_case** naming for all code and saved artifacts. Regarding security, all Supabase credentials must be stored in a local `.env` file that is never committed to version control, or managed via a secure secret provider in production environments.

---

## Deployment and Maintenance

Our CI/CD workflow is integrated with GitHub Actions, which automatically runs linting and unit tests on every push to the `main` branch to ensure that the production pipeline remains stable. For production use, batch predictions can be generated using the CLI interface by running `make predict INPUT=path/to/csv`, which will output the valuations for any list of domains provided in the input file.

---

## Maintainers
- **Growth Marketing Tools Team**
- **Primary Contact**: Vytautas Bunevicius (vytautas.bunevicius@nordsec.com)
