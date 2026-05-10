from __future__ import annotations

from src.config import PROCESSED_LOTTO_FILE
from src.data.load_data import load_lotto_source
from src.data.preprocess import preprocess_lotto_data, save_processed_lotto
from src.data.sync_lotto_html import sync_lotto_history_html
from src.data.sync_weather_context import (
    build_weather_context_from_cache,
    fetch_weather_observations,
    sync_weather_context,
)
from src.data.validate_data import (
    validate_number_ranges,
    validate_processed_columns,
    validate_unique_main_numbers,
)
from src.features.build_features import build_feature_dataset
from src.models.model_suite import run_full_model_suite


def run_data_pipeline(
    source: str = "excel",
    file_path=None,
    start_round: int = 1,
    end_round: int | None = None,
):
    raw_df = load_lotto_source(
        source=source,
        file_path=file_path,
        start_round=start_round,
        end_round=end_round,
    )
    clean_df = preprocess_lotto_data(raw_df)
    validate_processed_columns(clean_df)
    validate_number_ranges(clean_df)
    validate_unique_main_numbers(clean_df)
    save_processed_lotto(clean_df, PROCESSED_LOTTO_FILE)
    return PROCESSED_LOTTO_FILE


def run_feature_pipeline(window: int = 20, save: bool = True):
    return build_feature_dataset(window=window, save=save)


def run_sync_pipeline():
    return sync_lotto_history_html()


def run_weather_sync_pipeline(force: bool = False):
    return sync_weather_context(force=force)


def run_weather_fetch_pipeline(force: bool = False):
    return fetch_weather_observations(force=force)


def run_weather_build_pipeline():
    return build_weather_context_from_cache()


def run_model_pipeline(
    window: int = 20,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    backtest_initial_train_size: int = 600,
    backtest_test_size: int = 30,
    backtest_step_size: int = 30,
    holdout_model_names: list[str] | None = None,
    backtest_model_names: list[str] | None = None,
    include_backtest: bool = True,
    max_backtest_folds: int | None = None,
    feature_set_name: str = "base",
    save_base: bool = True,
    train_window_size: int | None = None,
):
    return run_full_model_suite(
        window=window,
        test_ratio=test_ratio,
        random_seed=random_seed,
        backtest_initial_train_size=backtest_initial_train_size,
        backtest_test_size=backtest_test_size,
        backtest_step_size=backtest_step_size,
        holdout_model_names=holdout_model_names,
        backtest_model_names=backtest_model_names,
        include_backtest=include_backtest,
        max_backtest_folds=max_backtest_folds,
        feature_set_name=feature_set_name,
        save_base=save_base,
        train_window_size=train_window_size,
    )
