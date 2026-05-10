from __future__ import annotations

import numpy as np


def _coerce_probability_output(proba) -> np.ndarray:
    if isinstance(proba, list):
        return np.column_stack([p[:, 1] for p in proba])

    proba = np.asarray(proba)

    if proba.ndim == 2:
        return proba

    if proba.ndim == 3 and proba.shape[-1] == 2:
        return proba[:, :, 1]

    raise ValueError(f"Unsupported probability output shape: {proba.shape}")


def build_ensemble_component_model(model_name: str, random_seed: int = 42):
    if model_name == "logistic_regression":
        return __import__(
            "src.models.train_baseline",
            fromlist=["build_logistic_regression_model"],
        ).build_logistic_regression_model(random_seed=random_seed)

    if model_name == "xgboost":
        return __import__(
            "src.models.train_xgboost",
            fromlist=["build_xgboost_model"],
        ).build_xgboost_model(random_seed=random_seed)

    if model_name == "classifier_chain":
        return __import__(
            "src.models.train_baseline",
            fromlist=["build_classifier_chain_model"],
        ).build_classifier_chain_model(random_seed=random_seed)

    raise ValueError(f"Unsupported ensemble component model: {model_name}")


class SoftVotingNumberEnsemble:
    def __init__(self, model_names: list[str], random_seed: int = 42, weights: list[float] | None = None):
        self.model_names = model_names
        self.random_seed = random_seed
        self.weights = np.asarray(weights if weights is not None else [1.0] * len(model_names), dtype=float)
        self.models_: list[tuple[str, object]] = []

    def fit(self, X, y):
        self.models_ = []
        for name in self.model_names:
            model = build_ensemble_component_model(name, random_seed=self.random_seed)
            model.fit(X, y)
            self.models_.append((name, model))
        return self

    def predict_proba(self, X):
        if not self.models_:
            raise RuntimeError("Ensemble must be fit before calling predict_proba.")

        probability_matrices = []
        for _, model in self.models_:
            probability_matrices.append(_coerce_probability_output(model.predict_proba(X)))
        normalized_weights = self.weights / self.weights.sum()
        return np.average(np.stack(probability_matrices, axis=0), axis=0, weights=normalized_weights)


def build_soft_voting_ensemble_model(random_seed: int = 42) -> SoftVotingNumberEnsemble:
    return SoftVotingNumberEnsemble(
        ["logistic_regression", "xgboost", "classifier_chain"],
        random_seed=random_seed,
        weights=[0.35, 0.40, 0.25],
    )
