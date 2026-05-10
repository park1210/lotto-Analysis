from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def build_random_forest_model(random_seed: int = 42) -> MultiOutputClassifier:
    return MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        )
    )
