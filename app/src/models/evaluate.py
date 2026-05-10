from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss

from src.models.predict import number_lists_to_binary_df


def row_to_numbers(row: pd.Series) -> list[int]:
    return sorted([int(col.split("_")[1]) for col, val in row.items() if val == 1])


def hits_per_draw(predicted_number_lists: list[list[int]], y_true: pd.DataFrame) -> list[int]:
    actual_numbers = y_true.apply(row_to_numbers, axis=1).reset_index(drop=True)
    return [
        len(set(pred) & set(actual))
        for pred, actual in zip(predicted_number_lists, actual_numbers)
    ]


def _compute_brier_score(y_true: pd.DataFrame, probability_matrix: np.ndarray | None) -> float:
    if probability_matrix is None:
        return float("nan")

    truth = y_true.to_numpy(dtype=float)
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.shape != truth.shape:
        raise ValueError(
            f"Probability matrix shape {probabilities.shape} does not match truth shape {truth.shape}."
        )

    return float(
        np.mean(
            [
                brier_score_loss(truth[:, col_idx], probabilities[:, col_idx])
                for col_idx in range(truth.shape[1])
            ]
        )
    )


def evaluate_number_predictions(
    model_name: str,
    predicted_number_lists: list[list[int]],
    y_true: pd.DataFrame,
    label_cols: list[str],
    probability_matrix: np.ndarray | None = None,
) -> dict:
    y_pred_binary = number_lists_to_binary_df(predicted_number_lists, label_cols=label_cols)
    actual_numbers = y_true.apply(row_to_numbers, axis=1).reset_index(drop=True)
    hit_scores = hits_per_draw(predicted_number_lists, y_true)

    draw_results = pd.DataFrame(
        {
            "model": model_name,
            "draw_index": np.arange(len(predicted_number_lists)),
            "predicted_numbers": [",".join(map(str, nums)) for nums in predicted_number_lists],
            "actual_numbers": [",".join(map(str, nums)) for nums in actual_numbers],
            "hit_count": hit_scores,
            "exact_match": (y_true.values == y_pred_binary.values).all(axis=1).astype(int),
        }
    )

    avg_hit = float(np.mean(hit_scores))
    precision_at_6 = avg_hit / 6.0
    recall_at_6 = avg_hit / 6.0

    return {
        "model": model_name,
        "subset_accuracy": accuracy_score(y_true.values, y_pred_binary.values),
        "number_level_accuracy": (y_true.values == y_pred_binary.values).mean(),
        "avg_hit": avg_hit,
        "hit_std": float(np.std(hit_scores)),
        "precision_at_6": precision_at_6,
        "recall_at_6": recall_at_6,
        "brier_score": _compute_brier_score(y_true, probability_matrix),
        "draw_results": draw_results,
    }


def flatten_probability_outputs(
    probability_matrix: np.ndarray,
    y_true: pd.DataFrame,
) -> pd.DataFrame:
    truth = y_true.to_numpy(dtype=int)
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.shape != truth.shape:
        raise ValueError(
            f"Probability matrix shape {probabilities.shape} does not match truth shape {truth.shape}."
        )

    return pd.DataFrame(
        {
            "predicted_probability": probabilities.reshape(-1),
            "actual": truth.reshape(-1),
        }
    )


def compute_calibration_table(
    probability_matrix: np.ndarray,
    y_true: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    flat = flatten_probability_outputs(probability_matrix, y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    flat["probability_bin"] = pd.cut(
        flat["predicted_probability"],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    calibration_table = (
        flat.groupby("probability_bin", observed=False)
        .agg(
            count=("actual", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_positive_rate=("actual", "mean"),
        )
        .reset_index()
    )
    return calibration_table


def compute_precision_at_k_curve(
    probability_matrix: np.ndarray,
    y_true: pd.DataFrame,
    ks: tuple[int, ...] = tuple(range(1, 11)),
) -> pd.DataFrame:
    truth = y_true.to_numpy(dtype=int)
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.shape != truth.shape:
        raise ValueError(
            f"Probability matrix shape {probabilities.shape} does not match truth shape {truth.shape}."
        )

    rows = []
    for k in ks:
        topk_idx = np.argsort(probabilities, axis=1)[:, -k:]
        hits = []
        for row_idx, indices in enumerate(topk_idx):
            hits.append(truth[row_idx, indices].sum() / k)
        rows.append({"k": k, "precision_at_k": float(np.mean(hits))})
    return pd.DataFrame(rows)
