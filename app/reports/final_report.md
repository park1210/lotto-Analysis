# Lotto Analysis Final Report

## Project Goal

This project analyzes historical Korean Lotto 6/45 data to answer three main questions:

1. Does the historical number distribution look materially different from a random process?
2. Can forecasting-oriented features be built from past draws without leakage?
3. Do any model and feature combinations outperform random-style baselines in a meaningful and repeatable way?

## Data Pipeline

The project uses a notebook-first workflow with canonical saved files.

- raw workbook: `app/data/raw/lotto_history_latest.xlsx`
- processed lotto file: `app/data/processed/lotto_cleaned.csv`
- weather metadata: `app/data/external/draw_metadata.csv`
- weather cache: `app/data/external/weather_observations.csv`
- draw-level weather context: `app/data/external/weather_draw_context.csv`

The current collection workflow is notebook-centered through shared helpers in `src.pipelines`.

- `run_sync_pipeline()` for lotto updates
- `run_weather_fetch_pipeline()` for weather cache collection
- `run_weather_build_pipeline()` for draw-level weather context construction
- `run_data_pipeline(source="excel")` for preprocessing
- `run_feature_pipeline(window=20)` for canonical base feature generation

## Exploratory Analysis

The exploratory stage focuses on:

- main-number frequency vs uniform expectation
- bonus-number frequency vs uniform expectation
- odd-even balance
- low-high balance
- total-sum distribution
- time-trend inspection
- pairwise number correlation

Saved EDA assets:

- `fig_01_main_number_frequency.png`
- `fig_02_bonus_number_frequency.png`
- `fig_03_odd_even_pattern.png`
- `fig_04_low_high_split.png`
- `fig_05_sum_distribution.png`

## Randomness Tests

The statistical testing stage compares real outcomes against Monte Carlo baselines.

Key checks:

- frequency comparison against simulated random draws
- chi-square test for uniformity
- KL divergence against simulated distributions
- consecutive-draw overlap comparison

Saved outputs:

- `fig_06_real_vs_random_frequency.png`
- `fig_06b_consecutive_draw_overlap.png`
- `table_04_randomness_test_summary.csv`

Interpretation: the project continues to support a conservative view that the historical data does not show strong evidence of large structural bias away from a random process.

## Feature Engineering

The baseline feature set is forecasting-oriented and currently includes:

- rolling recent-frequency features
- gap features measuring time since each number last appeared

These are aligned so each row uses only information available before the target draw.

The official model-facing feature registry now includes:

- `base`
- `base_plus_pattern`
- `base_plus_context`
- `full_feature_set`

This keeps the later modeling notebooks on consistent feature definitions.

## Calendar Context Extension

The calendar-context notebook derives:

- month
- season
- year
- day of month

It then compares:

- number-frequency distributions across context groups
- draw-pattern distributions such as `sum_main`, `odd_count`, and `low_count`

Saved outputs:

- `fig_11` to `fig_17`
- `table_10` to `table_14`

Interpretation: calendar context is useful as a descriptive analysis layer, but it has not yet produced strong evidence of a stable predictive advantage.

## Weather Extension

The weather pipeline builds draw-level weather context from KMA API Hub observations.

Saved analysis outputs:

- `fig_18` to `fig_25`
- `table_15` to `table_19`

Current measured weather status:

- `temp_at_draw`, `humidity_at_draw`, `wind_at_draw`, and `pressure_at_draw` are available for all `1217` draws
- `rain_flag` is now non-zero after revising the rain proxy
- `rain_flag = 94` draws, about `7.72%`
- `snow_flag = 6` draws, about `0.49%`
- daily weather aggregates are still unavailable in the saved run

Current statistical interpretation from `table_17_weather_pattern_test_summary.csv`:

- rain vs non-rain group tests do not show small p-values for `sum_main`, `odd_count`, or `low_count`
- snow vs non-snow tests also do not show small p-values
- temperature-bin and humidity-bin Kruskal tests do not show strong distribution shifts

Interpretation: weather context is currently more useful as exploratory context than as a strong direct explanatory driver of lotto outcomes.

## Weather-Aware Feature Modeling

`09_weather_feature_modeling.ipynb` compares whether calendar and weather variables improve the temporal baseline.

Saved outputs:

- `fig_26_weather_feature_holdout_comparison.png`
- `fig_27_weather_feature_backtest_comparison.png`
- `fig_28_weather_feature_backtest_trend.png`
- `table_20_weather_feature_holdout_summary.csv`
- `table_21_weather_feature_backtest_summary.csv`
- `table_22_weather_feature_backtest_results.csv`

Measured results:

- holdout `avg_hit`
  - `base = 0.8708`
  - `base_plus_calendar = 0.8583`
  - `base_plus_calendar_weather = 0.8458`
  - `calendar_weather_only = 0.8292`
- rolling backtest `mean_avg_hit`
  - `base = 0.8389`
  - `base_plus_calendar_weather = 0.8139`
  - `base_plus_calendar = 0.7944`
  - `calendar_weather_only = 0.7722`

Interpretation: the existing temporal baseline remains strongest. Calendar and weather context are better interpreted as explanatory covariates than as direct performance-improving predictors in the current setup.

## Baseline Modeling

The baseline modeling suite includes:

- `freq_heuristic`
- `gap_heuristic`
- `random_baseline`
- `logistic_regression`
- `random_forest`
- `xgboost`
- `classifier_chain`
- `mlp`
- `soft_voting_ensemble`

Saved outputs:

- `table_05_holdout_summary.csv`
- `table_06_backtest_summary.csv`
- `table_07_draw_level_results.csv`
- `table_08_run_metadata.csv`
- `table_09_model_output_paths.csv`
- model artifacts under `app/models/artifacts`

Measured baseline results:

- best holdout `avg_hit`: `classifier_chain = 0.9000`
- next best holdout `avg_hit`: `logistic_regression = 0.8708`
- best rolling backtest `mean_avg_hit`: `random_baseline = 0.8719`
- `logistic_regression` remains close in rolling backtest: `0.8684`
- all major runs still have `subset_accuracy = 0.0`

Interpretation: no model yet demonstrates a robust and large advantage over random-style baselines.

## Model-Family Comparison

`10_model_family_comparison.ipynb` compares four feature sets across multiple model families.

Feature sets:

- `base`
- `base_plus_pattern`
- `base_plus_context`
- `full_feature_set`

Model families:

- heuristic baselines
- linear model
- tree-based models
- classifier chain
- shallow neural network
- weighted soft-voting ensemble

Saved outputs:

- `fig_29_model_family_holdout_heatmap.png`
- `fig_30_model_family_backtest_heatmap.png`
- `fig_31_model_family_top_comparison.png`
- `fig_32_model_family_backtest_trend.png`
- `fig_33_model_family_calibration_curves.png`
- `fig_34_model_family_probability_histograms.png`
- `fig_35_model_family_precision_at_k.png`
- `table_23_model_family_holdout_summary.csv`
- `table_24_model_family_backtest_summary.csv`
- `table_25_model_family_full_results.csv`
- `table_26_model_family_calibration_summary.csv`
- `table_27_model_family_calibration_bins.csv`
- `table_28_model_family_precision_at_k.csv`

Measured results:

- best holdout combination: `base_plus_pattern + classifier_chain = 0.9042`
- second-best holdout combination: `base + classifier_chain = 0.9000`
- strong context-heavy combinations did not displace internal-pattern feature sets at the top of holdout ranking
- best backtest mean includes `random_baseline = 0.8567`
- `base_plus_context + mlp` matched that backtest mean by average score, but fold-level paths differ and the result should be interpreted cautiously

Interpretation:

- richer internal draw-pattern features appear more promising than weather/calendar context for prediction
- holdout improvements do not automatically transfer to robust rolling-backtest advantages
- calibration and ranking diagnostics are now available, but they still support a conservative view of the project¡¯s predictive ceiling

## Recent-Window Modeling

`11_recent_window_modeling.ipynb` compares recent-history training regimes.

Training regimes:

- `last_500`
- `last_300`
- `last_200`

Saved outputs:

- `fig_36_recent_window_holdout_heatmap.png`
- `fig_37_recent_window_backtest_heatmap.png`
- `fig_38_recent_window_top_comparison.png`
- `fig_39_recent_window_backtest_trend.png`
- `table_29_recent_window_holdout_summary.csv`
- `table_30_recent_window_backtest_summary.csv`
- `table_31_recent_window_full_results.csv`

Measured results:

- strongest holdout candidate: `base_plus_context + soft_voting_ensemble + last_500 = 0.9046`
- strongest shortlist candidates concentrate around `soft_voting_ensemble`, `mlp`, and `classifier_chain`
- among recent-only regimes, `last_500` is the most promising overall
- backtest gains remain moderate rather than decisive

Interpretation: recent-history restriction is not useless, but the evidence supports `last_500` much more than `last_300` or `last_200`.

## Final Candidate Selection

`12_final_candidate_selection.ipynb` merges notebooks 10 and 11 into a single shortlist and recommendation.

Saved outputs:

- `fig_40_final_candidate_scatter.png`
- `fig_41_final_candidate_score.png`
- `fig_42_final_candidate_by_model.png`
- `table_32_final_candidate_shortlist.csv`
- `table_33_final_candidate_full_table.csv`
- `table_34_final_recommendation.csv`

Final recommendation from `table_34_final_recommendation.csv`:

- `feature_set = full_feature_set`
- `model = soft_voting_ensemble`
- `training_regime = last_500`

Interpretation: the current project¡¯s best-balanced representative candidate is an ensemble model trained on the full feature registry but only the most recent 500 draws.

## Probability Ranking Analysis

`13_probability_ranking_analysis.ipynb` inspects the final candidate¡¯s probability behavior directly.

Saved outputs:

- `fig_43_probability_ranking_calibration.png`
- `fig_44_probability_ranking_density.png`
- `fig_45_probability_ranking_precision_at_k.png`
- `fig_46_probability_ranking_top10_heatmap.png`
- `table_35_probability_ranking_summary.csv`
- `table_36_probability_ranking_calibration_bins.csv`
- `table_37_probability_ranking_precision_at_k.csv`
- `table_38_probability_ranking_top10_frequency.csv`

Measured summary:

- `avg_hit = 0.8631`
- `precision_at_6 = 0.1438`
- `brier_score = 0.1952`
- `mean_positive_probability = 0.2774`
- `mean_negative_probability = 0.2680`

Interpretation: the model ranks numbers somewhat better than chance, but positive and negative classes are only modestly separated. This is more consistent with a usable ranking system than with a strongly discriminative exact-forecast model.

## Ticket Generation Strategy

`14_ticket_generation_strategy.ipynb` turns the final ranking model into practical ticket-construction strategies.

Saved outputs:

- `fig_47_ticket_strategy_avg_hit.png`
- `fig_48_ticket_strategy_hit_distribution.png`
- `fig_49_ticket_strategy_diversity.png`
- `table_39_ticket_strategy_full_results.csv`
- `table_40_ticket_strategy_summary.csv`
- `table_41_ticket_strategy_diversity_summary.csv`

Measured strategy results:

- `greedy_top6`
  - `avg_hit = 0.8631`
  - strongest single-ticket strategy
- `weighted_sampling`
  - `avg_hit = 0.8282`
  - `mean_pairwise_overlap = 1.43`
  - best diversity-oriented multi-ticket strategy
- `diversified_top_pool`
  - `avg_hit = 0.8382`
  - `mean_pairwise_overlap = 4.00`

Interpretation: if the goal is one representative ticket, `greedy_top6` is the strongest choice. If the goal is a more diversified portfolio of tickets, `weighted_sampling` is the best strategy among the tested options.

## Current Interpretation

At this stage, the project is best interpreted as a structured investigation into predictive limits and ranking quality rather than a claim that lotto outcomes can be strongly forecast.

The current evidence supports six conclusions:

- the historical distribution remains broadly compatible with random-style behavior
- temporal baseline features are still the most stable predictive foundation
- weather and calendar context help more as explanatory context than as direct predictive signal
- richer internal pattern features help more than external context features
- the most useful recent-only regime tested so far is `last_500`
- the project¡¯s strongest practical framing is a probability-ranking and ticket-strategy workflow, not exact draw prediction

## Recommended Practical Output

Based on the full sequence of notebooks, the project¡¯s current recommended setup is:

- final representative model:
  - `full_feature_set + soft_voting_ensemble + last_500`
- final single-ticket recommendation logic:
  - `greedy_top6`
- final multi-ticket diversified recommendation logic:
  - `weighted_sampling`

## Next Steps

Recommended follow-up work:

- tune the final candidate further around `soft_voting_ensemble + full_feature_set + last_500`
- add feature-importance and coefficient summaries so the final candidate is easier to interpret
- compare calibration and ranking quality against lighter top candidates such as `classifier_chain` and `mlp`
- improve or replace `diversified_top_pool` with a stronger diversity-aware generator
- add a final ¡°next draw recommendation¡± notebook or export layer built directly on the selected ranking model
