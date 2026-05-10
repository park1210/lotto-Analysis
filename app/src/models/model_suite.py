from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_RESULTS_FILE,
    BACKTEST_SUMMARY_FILE,
    HOLDOUT_DRAW_RESULTS_FILE,
    HOLDOUT_SUMMARY_FILE,
    MODEL_ARTIFACT_DIR,
    MODEL_OUTPUT_DIR,
    MODEL_RUN_METADATA_FILE,
)
from src.features import FEATURE_SET_ORDER, build_model_feature_bundle
from src.features.temporal_features import time_based_train_test_split
from src.models.evaluate import evaluate_number_predictions
from src.models.predict import probability_matrix_to_number_lists


LABEL_COLS = [f"y_{i}" for i in range(1, 46)]
FREQ_COLS = [f"freq_{i}" for i in range(1, 46)]
GAP_COLS = [f"gap_{i}" for i in range(1, 46)]
HEURISTIC_MODEL_NAMES = ["freq_heuristic", "gap_heuristic", "random_baseline"]
ML_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "classifier_chain", "mlp", "soft_voting_ensemble"]
ALL_MODEL_NAMES = HEURISTIC_MODEL_NAMES + ML_MODEL_NAMES
DEFAULT_HOLDOUT_MODEL_NAMES = [
    "freq_heuristic",
    "gap_heuristic",
    "random_baseline",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "classifier_chain",
    "soft_voting_ensemble",
]
DEFAULT_BACKTEST_MODEL_NAMES = [
    "freq_heuristic",
    "gap_heuristic",
    "random_baseline",
    "logistic_regression",
    "soft_voting_ensemble",
]


def prepare_model_data(
    window: int = 20,
    test_ratio: float = 0.2,
    feature_set_name: str = "base",
    save_base: bool = True,
):
    feature_bundle = build_model_feature_bundle(window=window, save_base=save_base)

    if feature_set_name not in feature_bundle.feature_sets:
        raise ValueError(
            f"Unsupported feature set '{feature_set_name}'. Available: {sorted(feature_bundle.feature_sets)}"
        )

    X = feature_bundle.feature_sets[feature_set_name].reset_index(drop=True)
    y = feature_bundle.y.reset_index(drop=True)
    split_data = time_based_train_test_split(X, y, test_ratio=test_ratio)

    return {
        "X": X,
        "y": y,
        "X_train": split_data["X_train"],
        "X_test": split_data["X_test"],
        "y_train": split_data["y_train"],
        "y_test": split_data["y_test"],
        "split_idx": split_data["split_idx"],
        "feature_bundle": feature_bundle,
        "feature_set_name": feature_set_name,
    }


def get_probability_matrix(model, X_frame: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_frame)

    if isinstance(proba, list):
        return np.column_stack([p[:, 1] for p in proba])

    proba = np.asarray(proba)

    if proba.ndim == 2:
        return proba

    if proba.ndim == 3 and proba.shape[-1] == 2:
        return proba[:, :, 1]

    raise ValueError(f"Unsupported probability output shape: {proba.shape}")


def evaluate_probability_model(model_name: str, model, X_frame: pd.DataFrame, y_true: pd.DataFrame) -> dict:
    probability_matrix = get_probability_matrix(model, X_frame)
    predicted_number_lists = probability_matrix_to_number_lists(probability_matrix)
    return evaluate_number_predictions(
        model_name,
        predicted_number_lists,
        y_true,
        label_cols=LABEL_COLS,
        probability_matrix=probability_matrix,
    )


def frequency_heuristic_predictions(X_frame: pd.DataFrame) -> list[list[int]]:
    return probability_matrix_to_number_lists(X_frame[FREQ_COLS].to_numpy())


def gap_heuristic_predictions(X_frame: pd.DataFrame) -> list[list[int]]:
    return probability_matrix_to_number_lists(X_frame[GAP_COLS].to_numpy())


def random_predictions(n_rows: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        for _ in range(n_rows)
    ]


def build_model_builders(random_seed: int) -> dict[str, object]:
    return {
        "logistic_regression": lambda: __import__(
            "src.models.train_baseline",
            fromlist=["build_logistic_regression_model"],
        ).build_logistic_regression_model(random_seed=random_seed),
        "random_forest": lambda: __import__(
            "src.models.train_random_forest",
            fromlist=["build_random_forest_model"],
        ).build_random_forest_model(random_seed=random_seed),
        "xgboost": lambda: __import__(
            "src.models.train_xgboost",
            fromlist=["build_xgboost_model"],
        ).build_xgboost_model(random_seed=random_seed),
        "classifier_chain": lambda: __import__(
            "src.models.train_baseline",
            fromlist=["build_classifier_chain_model"],
        ).build_classifier_chain_model(random_seed=random_seed),
        "mlp": lambda: __import__(
            "src.models.train_mlp",
            fromlist=["build_scaled_mlp_model"],
        ).build_scaled_mlp_model(random_seed=random_seed),
        "soft_voting_ensemble": lambda: __import__(
            "src.models.train_ensemble",
            fromlist=["build_soft_voting_ensemble_model"],
        ).build_soft_voting_ensemble_model(random_seed=random_seed),
    }


def resolve_model_names(model_names: list[str] | None, default_model_names: list[str]) -> list[str]:
    if model_names is None:
        return default_model_names.copy()

    invalid_names = sorted(set(model_names) - set(ALL_MODEL_NAMES))
    if invalid_names:
        raise ValueError(f"Unsupported model names: {invalid_names}")

    resolved = []
    for model_name in model_names:
        if model_name not in resolved:
            resolved.append(model_name)
    return resolved


def run_holdout_experiments(
    data: dict[str, pd.DataFrame],
    random_seed: int = 42,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    selected_model_names = resolve_model_names(model_names, DEFAULT_HOLDOUT_MODEL_NAMES)

    results = []
    builders = build_model_builders(random_seed)
    trained_models: dict[str, object] = {}

    for model_name in selected_model_names:
        if model_name == "freq_heuristic":
            results.append(
                evaluate_number_predictions(
                    model_name,
                    frequency_heuristic_predictions(X_test),
                    y_test,
                    label_cols=LABEL_COLS,
                )
            )
        elif model_name == "gap_heuristic":
            results.append(
                evaluate_number_predictions(
                    model_name,
                    gap_heuristic_predictions(X_test),
                    y_test,
                    label_cols=LABEL_COLS,
                )
            )
        elif model_name == "random_baseline":
            results.append(
                evaluate_number_predictions(
                    model_name,
                    random_predictions(len(X_test), random_seed),
                    y_test,
                    label_cols=LABEL_COLS,
                )
            )
        else:
            model = builders[model_name]()
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            results.append(evaluate_probability_model(model_name, model, X_test, y_test))

    holdout_summary = pd.DataFrame(
        [
            {
                "model": result["model"],
                "subset_accuracy": result["subset_accuracy"],
                "number_level_accuracy": result["number_level_accuracy"],
                "avg_hit": result["avg_hit"],
                "hit_std": result["hit_std"],
                "precision_at_6": result["precision_at_6"],
                "recall_at_6": result["recall_at_6"],
                "brier_score": result["brier_score"],
            }
            for result in results
        ]
    ).sort_values(["avg_hit", "precision_at_6", "number_level_accuracy"], ascending=False)

    holdout_draw_results = pd.concat([result["draw_results"] for result in results], ignore_index=True)
    return holdout_summary, holdout_draw_results, trained_models


def evaluate_model_on_split(
    model_name: str,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.DataFrame,
    X_test_fold: pd.DataFrame,
    y_test_fold: pd.DataFrame,
    fold_idx: int,
    random_seed: int,
) -> dict:
    if model_name == "freq_heuristic":
        result = evaluate_number_predictions(model_name, frequency_heuristic_predictions(X_test_fold), y_test_fold, label_cols=LABEL_COLS)
    elif model_name == "gap_heuristic":
        result = evaluate_number_predictions(model_name, gap_heuristic_predictions(X_test_fold), y_test_fold, label_cols=LABEL_COLS)
    elif model_name == "random_baseline":
        result = evaluate_number_predictions(
            model_name,
            random_predictions(len(X_test_fold), seed=random_seed + fold_idx),
            y_test_fold,
            label_cols=LABEL_COLS,
        )
    else:
        model = build_model_builders(random_seed)[model_name]()
        model.fit(X_train_fold, y_train_fold)
        result = evaluate_probability_model(model_name, model, X_test_fold, y_test_fold)

    return {
        "fold": fold_idx,
        "model": result["model"],
        "subset_accuracy": result["subset_accuracy"],
        "number_level_accuracy": result["number_level_accuracy"],
        "avg_hit": result["avg_hit"],
        "precision_at_6": result["precision_at_6"],
        "recall_at_6": result["recall_at_6"],
        "brier_score": result["brier_score"],
    }


def run_rolling_backtest(
    data: dict[str, pd.DataFrame],
    initial_train_size: int = 600,
    test_size: int = 30,
    step_size: int = 30,
    random_seed: int = 42,
    model_names: list[str] | None = None,
    max_folds: int | None = None,
    train_window_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data["X"]
    y = data["y"]
    selected_model_names = resolve_model_names(model_names, DEFAULT_BACKTEST_MODEL_NAMES)

    rows = []
    fold_idx = 0

    for train_end in range(initial_train_size, len(X) - test_size + 1, step_size):
        if max_folds is not None and fold_idx >= max_folds:
            break

        test_end = train_end + test_size
        train_start = max(0, train_end - train_window_size) if train_window_size is not None else 0
        X_train_fold = X.iloc[train_start:train_end].reset_index(drop=True)
        y_train_fold = y.iloc[train_start:train_end].reset_index(drop=True)
        X_test_fold = X.iloc[train_end:test_end].reset_index(drop=True)
        y_test_fold = y.iloc[train_end:test_end].reset_index(drop=True)

        for model_name in selected_model_names:
            rows.append(
                evaluate_model_on_split(
                    model_name,
                    X_train_fold,
                    y_train_fold,
                    X_test_fold,
                    y_test_fold,
                    fold_idx,
                    random_seed,
                )
            )

        fold_idx += 1

    backtest_results = pd.DataFrame(rows)
    if backtest_results.empty:
        backtest_summary = pd.DataFrame(
            columns=[
                "model",
                "folds",
                "mean_subset_accuracy",
                "mean_number_level_accuracy",
                "mean_avg_hit",
                "mean_precision_at_6",
                "mean_recall_at_6",
                "mean_brier_score",
                "std_avg_hit",
            ]
        )
    else:
        backtest_summary = backtest_results.groupby("model", as_index=False).agg(
            folds=("fold", "nunique"),
            mean_subset_accuracy=("subset_accuracy", "mean"),
            mean_number_level_accuracy=("number_level_accuracy", "mean"),
            mean_avg_hit=("avg_hit", "mean"),
            mean_precision_at_6=("precision_at_6", "mean"),
            mean_recall_at_6=("recall_at_6", "mean"),
            mean_brier_score=("brier_score", "mean"),
            std_avg_hit=("avg_hit", "std"),
        ).sort_values("mean_avg_hit", ascending=False)

    return backtest_results, backtest_summary


def save_experiment_outputs(
    holdout_summary: pd.DataFrame,
    holdout_draw_results: pd.DataFrame,
    backtest_results: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    metadata: dict,
    trained_models: dict[str, object],
    output_dir: Path = MODEL_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    holdout_summary.to_csv(HOLDOUT_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    holdout_draw_results.to_csv(HOLDOUT_DRAW_RESULTS_FILE, index=False, encoding="utf-8-sig")
    backtest_results.to_csv(BACKTEST_RESULTS_FILE, index=False, encoding="utf-8-sig")
    backtest_summary.to_csv(BACKTEST_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    MODEL_RUN_METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifact_paths: dict[str, Path] = {}
    for model_name, model in trained_models.items():
        artifact_path = MODEL_ARTIFACT_DIR / f"{model_name}.joblib"
        joblib.dump(model, artifact_path)
        artifact_paths[model_name] = artifact_path

    paths = {
        "holdout_summary": HOLDOUT_SUMMARY_FILE,
        "holdout_draw_results": HOLDOUT_DRAW_RESULTS_FILE,
        "backtest_results": BACKTEST_RESULTS_FILE,
        "backtest_summary": BACKTEST_SUMMARY_FILE,
        "metadata": MODEL_RUN_METADATA_FILE,
    }
    for model_name, artifact_path in artifact_paths.items():
        paths[f"artifact_{model_name}"] = artifact_path
    return paths


def run_full_model_suite(
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
) -> dict[str, object]:
    data = prepare_model_data(
        window=window,
        test_ratio=test_ratio,
        feature_set_name=feature_set_name,
        save_base=save_base,
    )
    resolved_holdout_model_names = resolve_model_names(holdout_model_names, DEFAULT_HOLDOUT_MODEL_NAMES)
    resolved_backtest_model_names = resolve_model_names(backtest_model_names, DEFAULT_BACKTEST_MODEL_NAMES)

    holdout_summary, holdout_draw_results, trained_models = run_holdout_experiments(
        data,
        random_seed=random_seed,
        model_names=resolved_holdout_model_names,
    )

    if include_backtest:
        backtest_results, backtest_summary = run_rolling_backtest(
            data,
            initial_train_size=backtest_initial_train_size,
            test_size=backtest_test_size,
            step_size=backtest_step_size,
            random_seed=random_seed,
            model_names=resolved_backtest_model_names,
            max_folds=max_backtest_folds,
            train_window_size=train_window_size,
        )
    else:
        backtest_results = pd.DataFrame(columns=[
            "fold", "model", "subset_accuracy", "number_level_accuracy", "avg_hit", "precision_at_6", "recall_at_6", "brier_score"
        ])
        backtest_summary = pd.DataFrame(
            columns=[
                "model",
                "folds",
                "mean_subset_accuracy",
                "mean_number_level_accuracy",
                "mean_avg_hit",
                "mean_precision_at_6",
                "mean_recall_at_6",
                "mean_brier_score",
                "std_avg_hit",
            ]
        )

    metadata = {
        "window": window,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "feature_set_name": feature_set_name,
        "available_feature_sets": FEATURE_SET_ORDER,
        "holdout_model_names": resolved_holdout_model_names,
        "backtest_model_names": resolved_backtest_model_names if include_backtest else [],
        "include_backtest": include_backtest,
        "backtest_initial_train_size": backtest_initial_train_size,
        "backtest_test_size": backtest_test_size,
        "backtest_step_size": backtest_step_size,
        "max_backtest_folds": max_backtest_folds,
        "n_rows": len(data["X"]),
        "n_features": int(data["X"].shape[1]),
        "n_train_rows": len(data["X_train"]),
        "n_test_rows": len(data["X_test"]),
        "train_window_size": train_window_size,
    }

    paths = save_experiment_outputs(
        holdout_summary,
        holdout_draw_results,
        backtest_results,
        backtest_summary,
        metadata,
        trained_models,
    )

    return {
        "data": data,
        "holdout_summary": holdout_summary,
        "holdout_draw_results": holdout_draw_results,
        "backtest_results": backtest_results,
        "backtest_summary": backtest_summary,
        "metadata": metadata,
        "paths": paths,
    }
