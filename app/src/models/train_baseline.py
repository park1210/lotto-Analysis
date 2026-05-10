from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_scaled_logistic_estimator(random_seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def build_logistic_regression_model(random_seed: int = 42) -> MultiOutputClassifier:
    return MultiOutputClassifier(_build_scaled_logistic_estimator(random_seed=random_seed))


def build_classifier_chain_model(random_seed: int = 42) -> ClassifierChain:
    return ClassifierChain(
        _build_scaled_logistic_estimator(random_seed=random_seed),
        random_state=random_seed,
    )
