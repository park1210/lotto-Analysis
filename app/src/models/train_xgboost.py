from __future__ import annotations

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


DEFAULT_SCALE_POS_WEIGHT = 39 / 6


def build_xgboost_model(random_seed: int = 42) -> MultiOutputClassifier:
    return MultiOutputClassifier(
        XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=DEFAULT_SCALE_POS_WEIGHT,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=1,
            verbosity=0,
        )
    )
