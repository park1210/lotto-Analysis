from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
DEFAULT_PATTERN_SOURCE_COLS = ["sum_main", "odd_count", "low_count", "max_consecutive_run"]


def _max_consecutive_run(values: list[int]) -> int:
    sorted_values = sorted(int(v) for v in values)
    if not sorted_values:
        return 0

    max_run = 1
    current_run = 1
    for prev, curr in zip(sorted_values, sorted_values[1:]):
        if curr - prev == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def add_draw_pattern_columns(
    clean_df: pd.DataFrame,
    number_cols: list[str] | None = None,
) -> pd.DataFrame:
    number_cols = number_cols or NUMBER_COLS
    draw_patterns = clean_df.copy()
    draw_patterns["sum_main"] = draw_patterns[number_cols].sum(axis=1)
    draw_patterns["odd_count"] = draw_patterns[number_cols].apply(lambda row: sum(int(v) % 2 for v in row), axis=1)
    draw_patterns["low_count"] = draw_patterns[number_cols].apply(lambda row: sum(int(v) <= 22 for v in row), axis=1)
    draw_patterns["max_consecutive_run"] = draw_patterns[number_cols].apply(lambda row: _max_consecutive_run(row.tolist()), axis=1)
    return draw_patterns


def rolling_feature_block(
    df: pd.DataFrame,
    cols: list[str],
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        shifted = df[col].shift(1)
        out[f"{col}_prev"] = shifted
        for window in windows:
            out[f"{col}_mean_{window}"] = shifted.rolling(window).mean()
            out[f"{col}_std_{window}"] = shifted.rolling(window).std()
    return out


def build_pattern_feature_block(
    clean_df: pd.DataFrame,
    align_window: int = 20,
    rolling_windows: tuple[int, ...] = (5, 10, 20),
    source_cols: list[str] | None = None,
) -> pd.DataFrame:
    source_cols = source_cols or DEFAULT_PATTERN_SOURCE_COLS
    draw_patterns = add_draw_pattern_columns(clean_df)
    pattern_df = rolling_feature_block(draw_patterns, source_cols, windows=rolling_windows)
    pattern_df = pattern_df.loc[align_window:].reset_index(drop=True)
    return pattern_df.fillna(pattern_df.median(numeric_only=True)).fillna(0)
