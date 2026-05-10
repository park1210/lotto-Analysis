# Lotto Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pytest](https://img.shields.io/badge/Test-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-006400?style=flat-square)
![GitHub](https://img.shields.io/badge/Version%20Control-GitHub-181717?style=flat-square&logo=github&logoColor=white)

This project studies Korean Lotto 6/45 history through a notebook-first workflow. The current scope covers:

- raw lotto sync and preprocessing
- exploratory and statistical analysis
- calendar and weather context analysis
- baseline and extended model comparison
- report-ready figure and table exports

## Project Status

The project currently includes notebook steps `01` through `10`.

- `01_data_collection.ipynb`: data collection overview
- `02_eda.ipynb`: exploratory visualization
- `03_statistical_tests.ipynb`: randomness-oriented tests
- `04_feature_engineering.ipynb`: temporal feature construction
- `05_model_baseline.ipynb`: baseline model training and artifact export
- `06_model_evaluation.ipynb`: evaluation plots and summaries
- `07_contextual_analysis.ipynb`: calendar context analysis
- `08_weather_context_analysis.ipynb`: weather context analysis
- `09_weather_feature_modeling.ipynb`: weather-aware feature comparison
- `10_model_family_comparison.ipynb`: feature-set and model-family comparison

## Current Findings

### Baseline Modeling

From the baseline model suite:

- best holdout `avg_hit`: `classifier_chain = 0.9000`
- next best holdout `avg_hit`: `logistic_regression = 0.8708`
- best rolling backtest `mean_avg_hit`: `random_baseline = 0.8719`
- `subset_accuracy` remains `0.0` across saved runs

This means the current project still does not show strong evidence that learned models reliably outperform a random-style baseline in a robust way.

### Weather Context

The current weather context file now contains non-zero rain detection.

- `rain_flag`: `94` draws, about `7.72%`
- `snow_flag`: `6` draws, about `0.49%`

Current weather-group tests do not show strong evidence that rain, snow, temperature bins, or humidity bins materially shift draw patterns.

### Weather-Aware Feature Modeling

From `09_weather_feature_modeling.ipynb`:

- holdout `avg_hit`
  - `base = 0.8708`
  - `base_plus_calendar = 0.8583`
  - `base_plus_calendar_weather = 0.8458`
  - `calendar_weather_only = 0.8292`
- backtest `mean_avg_hit`
  - `base = 0.8389`
  - `base_plus_calendar_weather = 0.8139`
  - `base_plus_calendar = 0.7944`
  - `calendar_weather_only = 0.7722`

Interpretation: calendar and weather variables are currently more useful as explanatory context than as performance-improving predictors.

### Model-Family Comparison

From `10_model_family_comparison.ipynb`:

- best holdout combination: `base_plus_pattern + classifier_chain = 0.9042`
- next best holdout combination: `base + classifier_chain = 0.9000`
- best backtest combinations are still close to heuristic or weakly-structured baselines
- `base_plus_context + mlp` matched the top backtest mean by average score, but fold-level paths differ and the model should be interpreted cautiously because MLP remains sensitive to scaling and small-sample instability

Interpretation: richer internal pattern features help more than calendar/weather context, and robust superiority over random-style baselines remains limited.

## Run Guide

This project is notebook-first. The preferred workflow is:

1. start Docker and open Jupyter
2. use shared pipeline helpers from `src.pipelines` inside notebooks
3. keep raw collection, preprocessing, feature generation, and modeling reproducible through those shared functions
4. build official feature sets through `src.features` rather than reassembling them ad hoc inside notebooks
5. use notebooks for orchestration, interpretation, and export

### Start Docker

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Then open Jupyter:

- `http://localhost:8888`

### Standard Notebook Bootstrap

Every notebook now resolves the app root in the same way and can import from `src.*`.

Typical first-use imports inside notebooks:

```python
from src.pipelines import (
    run_sync_pipeline,
    run_data_pipeline,
    run_feature_pipeline,
    run_weather_fetch_pipeline,
    run_weather_build_pipeline,
    run_weather_sync_pipeline,
    run_model_pipeline,
)
```

### Lotto Sync in Notebook

Canonical raw workbook:

- `app/data/raw/lotto_history_latest.xlsx`

Run from a notebook cell:

```python
run_sync_pipeline()
```

Behavior:

- reads the last saved round in the workbook
- fetches only later HTML rounds
- stops immediately if the next round is unavailable
- updates the canonical workbook in place

### Data Preprocessing in Notebook

Processed output:

- `app/data/processed/lotto_cleaned.csv`

Run from a notebook cell:

```python
run_data_pipeline(source="excel")
```

If you want to point the preprocessing pipeline at a different workbook path from a notebook, you can also pass `file_path=...`.

### Feature Generation in Notebook

Feature output:

- `app/data/processed/lotto_features.csv`

Run from a notebook cell:

```python
feature_df = run_feature_pipeline(window=20)
feature_df.head()
```

For model-facing experiments, prefer the shared feature registry from `src.features` so notebook runs use the same feature-set definitions as the saved artifacts.

Shared feature structure now lives in `src.features`:

- `build_feature_dataset(...)`: canonical base `freq + gap` dataset
- `build_model_feature_bundle(...)`: aligned labels plus official feature sets
- `describe_feature_sets(...)`: quick summary of the current feature registry

The current official feature-set names are:

- `base`
- `base_plus_pattern`
- `base_plus_context`
- `full_feature_set`

This keeps Notebook 04, 09, and 10 on the same feature definitions.

### Weather Sync in Notebook

Canonical weather files:

- `app/data/external/draw_metadata.csv`
- `app/data/external/weather_observations.csv`
- `app/data/external/weather_draw_context.csv`

Environment setup:

```env
KMA_AUTH_KEY=your-issued-key
```

Recommended workflow from notebook cells:

1. fetch raw weather observations

```python
weather_fetch_bundle = run_weather_fetch_pipeline()
```

2. build draw-level weather context

```python
weather_build_bundle = run_weather_build_pipeline()
```

Optional combined helper:

```python
weather_sync_bundle = run_weather_sync_pipeline()
```

Notes:

- prefer `run_weather_fetch_pipeline()` for incremental collection over multiple runs
- prefer `run_weather_build_pipeline()` after fetching because it is local and lighter than remote collection
- avoid force-refetch workflows unless you intentionally want to ignore the cached weather observations

### Modeling in Notebook

Model outputs:

- `app/models/results/`
- `app/models/artifacts/`

Run from a notebook cell:

```python
results = run_model_pipeline(
    window=20,
    test_ratio=0.2,
    random_seed=42,
    backtest_initial_train_size=600,
    backtest_test_size=30,
    backtest_step_size=30,
    feature_set_name="base",
)
```

You can switch `feature_set_name` to `base_plus_pattern`, `base_plus_context`, or `full_feature_set` when you want Notebook 05 to run the official suite on a richer feature block.

You can also set `train_window_size=200` (or `300`, `500`) to test whether recent-only training is more stable than full-history training.

## Notebook Guidance

Recommended notebook usage:

- use stored files rather than embedding long live-fetch logic directly in analysis cells
- keep collection and preprocessing steps callable through `src.pipelines`
- import reusable logic from `src.analysis`, `src.features`, `src.models`, and `src.visualization`
- weather notebooks should read `weather_draw_context.csv` after `run_weather_build_pipeline()`
- weather exploration notebook: `app/notebooks/08_weather_context_analysis.ipynb`
- weather-aware feature notebook: `app/notebooks/09_weather_feature_modeling.ipynb`
- model-family comparison notebook: `app/notebooks/10_model_family_comparison.ipynb`

## Repository Layout

```text
002_lottoAnalysis/
?? app/
?  ?? data/
?  ?  ?? raw/
?  ?  ?? processed/
?  ?  ?? external/
?  ?? notebooks/
?  ?? reports/
?  ?? src/
?  ?? tests/
?? docker/
?? Makefile
?? README.md
```

## Weather Data Source Policy

Weather data in `app/data/external/` comes from KMA API Hub.

Relevant references:

- KMA API Hub info: https://apihub.kma.go.kr/apiInfo.do
- KMA copyright policy: https://www.kma.go.kr/kma/guide/copyright.jsp

Practical guidance for this repository:

- derived weather CSV outputs are likely acceptable to share when source attribution is preserved
- keep attribution in this README and `app/data/external/README.md`
- never commit `.env` or any private API key
- the safest minimal-publication approach is to prioritize `draw_metadata.csv` and `weather_draw_context.csv`

## Suggested Next Analyses

The most promising follow-up work is now inside the existing lotto-history feature space rather than from adding more external covariates.

Recommended directions:

- rerun `10_model_family_comparison.ipynb` with the shared feature registry, imbalance-aware linear models, and the weighted soft-voting ensemble
- test recent-round-only modeling windows such as the most recent `200`, `300`, or `500` draws in a dedicated follow-up notebook
- add lagged weather-regime features such as dry/wet streak length, recent mean temperature, and short humidity-trend features
- compare extreme-weather subsets against matched non-extreme draws
- extend model-family comparison with stronger feature-ranking diagnostics and coefficient/importance summaries
