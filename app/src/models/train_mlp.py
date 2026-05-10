from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_scaled_mlp_model(random_seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=5e-4,
                    learning_rate_init=5e-4,
                    max_iter=1200,
                    early_stopping=True,
                    n_iter_no_change=25,
                    validation_fraction=0.1,
                    random_state=random_seed,
                ),
            ),
        ]
    )
