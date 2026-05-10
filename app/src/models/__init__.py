from __future__ import annotations

__all__ = [
    "evaluate_number_predictions",
    "hits_per_draw",
    "row_to_numbers",
    "run_full_model_suite",
    "number_lists_to_binary_df",
    "probability_matrix_to_number_lists",
    "select_top6",
    "build_classifier_chain_model",
    "build_logistic_regression_model",
    "build_soft_voting_ensemble_model",
    "build_scaled_mlp_model",
    "build_random_forest_model",
    "build_xgboost_model",
]


def __getattr__(name: str):
    if name in {"evaluate_number_predictions", "hits_per_draw", "row_to_numbers"}:
        from .evaluate import evaluate_number_predictions, hits_per_draw, row_to_numbers
        return {
            "evaluate_number_predictions": evaluate_number_predictions,
            "hits_per_draw": hits_per_draw,
            "row_to_numbers": row_to_numbers,
        }[name]

    if name == "run_full_model_suite":
        from .model_suite import run_full_model_suite
        return run_full_model_suite

    if name in {"number_lists_to_binary_df", "probability_matrix_to_number_lists", "select_top6"}:
        from .predict import number_lists_to_binary_df, probability_matrix_to_number_lists, select_top6
        return {
            "number_lists_to_binary_df": number_lists_to_binary_df,
            "probability_matrix_to_number_lists": probability_matrix_to_number_lists,
            "select_top6": select_top6,
        }[name]

    if name in {"build_classifier_chain_model", "build_logistic_regression_model"}:
        from .train_baseline import build_classifier_chain_model, build_logistic_regression_model
        return {
            "build_classifier_chain_model": build_classifier_chain_model,
            "build_logistic_regression_model": build_logistic_regression_model,
        }[name]

    if name == "build_soft_voting_ensemble_model":
        from .train_ensemble import build_soft_voting_ensemble_model
        return build_soft_voting_ensemble_model

    if name == "build_scaled_mlp_model":
        from .train_mlp import build_scaled_mlp_model
        return build_scaled_mlp_model

    if name == "build_random_forest_model":
        from .train_random_forest import build_random_forest_model
        return build_random_forest_model

    if name == "build_xgboost_model":
        from .train_xgboost import build_xgboost_model
        return build_xgboost_model

    raise AttributeError(name)
