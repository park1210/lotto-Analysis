from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import PROCESSED_LOTTO_FILE
from src.features.build_features import build_feature_dataset
from src.features.context_features import build_context_feature_block, load_weather_context
from src.features.label_features import build_label_frame
from src.features.pattern_features import build_pattern_feature_block
from src.features.temporal_features import align_features_and_labels

FEATURE_SET_ORDER = [
    "base",
    "base_plus_pattern",
    "base_plus_context",
    "full_feature_set",
]

FEATURE_SET_DESCRIPTIONS = {
    "base": "Rolling frequency and gap features only.",
    "base_plus_pattern": "Base features plus draw-level internal pattern history.",
    "base_plus_context": "Base features plus calendar and weather context.",
    "full_feature_set": "Base, internal pattern, and context features combined.",
}


@dataclass
class ModelFeatureBundle:
    clean_df: pd.DataFrame
    feature_df: pd.DataFrame
    label_df: pd.DataFrame
    y: pd.DataFrame
    target_rounds: pd.Series
    base: pd.DataFrame
    pattern: pd.DataFrame
    context: pd.DataFrame
    feature_sets: dict[str, pd.DataFrame]
    base_feature_cols: list[str]
    pattern_feature_cols: list[str]
    calendar_feature_cols: list[str]
    weather_feature_cols: list[str]


def describe_feature_sets(bundle: ModelFeatureBundle) -> pd.DataFrame:
    rows = []
    for feature_set_name in FEATURE_SET_ORDER:
        frame = bundle.feature_sets[feature_set_name]
        rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": int(frame.shape[1]),
                "description": FEATURE_SET_DESCRIPTIONS[feature_set_name],
            }
        )
    return pd.DataFrame(rows)


def build_model_feature_bundle(
    window: int = 20,
    save_base: bool = True,
    weather_df: pd.DataFrame | None = None,
    pattern_windows: tuple[int, ...] = (5, 10, 20),
) -> ModelFeatureBundle:
    clean_df = pd.read_csv(PROCESSED_LOTTO_FILE, low_memory=False)
    feature_df = build_feature_dataset(window=window, save=save_base)
    label_df = build_label_frame(clean_df)
    base, y = align_features_and_labels(feature_df, label_df, window=window)
    target_rounds = clean_df.loc[window:, "round"].reset_index(drop=True)

    pattern_df = build_pattern_feature_block(clean_df, align_window=window, rolling_windows=pattern_windows)

    if weather_df is None:
        weather_df = load_weather_context()
    context_df, calendar_feature_cols, weather_feature_cols = build_context_feature_block(target_rounds, weather_df=weather_df)
    X_context = context_df.drop(columns=["round"]).reset_index(drop=True)

    feature_sets = {
        "base": base.reset_index(drop=True),
        "base_plus_pattern": pd.concat([base.reset_index(drop=True), pattern_df.reset_index(drop=True)], axis=1),
        "base_plus_context": pd.concat([base.reset_index(drop=True), X_context], axis=1),
        "full_feature_set": pd.concat([base.reset_index(drop=True), pattern_df.reset_index(drop=True), X_context], axis=1),
    }

    return ModelFeatureBundle(
        clean_df=clean_df,
        feature_df=feature_df,
        label_df=label_df,
        y=y.reset_index(drop=True),
        target_rounds=target_rounds,
        base=base.reset_index(drop=True),
        pattern=pattern_df.reset_index(drop=True),
        context=context_df.reset_index(drop=True),
        feature_sets=feature_sets,
        base_feature_cols=base.columns.tolist(),
        pattern_feature_cols=pattern_df.columns.tolist(),
        calendar_feature_cols=calendar_feature_cols,
        weather_feature_cols=weather_feature_cols,
    )
