from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WEATHER_CONTEXT_FILE


def load_weather_context(weather_path=WEATHER_CONTEXT_FILE) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_path, low_memory=False)
    weather_df["draw_date"] = pd.to_datetime(weather_df["draw_date"], errors="coerce")
    weather_df["draw_datetime_used"] = pd.to_datetime(weather_df["draw_datetime_used"], errors="coerce")
    return weather_df


def prepare_calendar_weather_frame(weather_df: pd.DataFrame) -> pd.DataFrame:
    calendar_weather = weather_df.copy()
    calendar_weather["month"] = calendar_weather["draw_date"].dt.month
    calendar_weather["year"] = calendar_weather["draw_date"].dt.year
    calendar_weather["day_of_month"] = calendar_weather["draw_date"].dt.day
    calendar_weather["season"] = pd.Categorical(
        np.select(
            [
                calendar_weather["month"].isin([12, 1, 2]),
                calendar_weather["month"].isin([3, 4, 5]),
                calendar_weather["month"].isin([6, 7, 8]),
            ],
            ["winter", "spring", "summer"],
            default="autumn",
        ),
        categories=["spring", "summer", "autumn", "winter"],
    )

    numeric_weather_cols = [
        "temp_at_draw",
        "humidity_at_draw",
        "wind_at_draw",
        "pressure_at_draw",
        "precip_at_draw",
        "precip_day_cumulative_at_draw",
        "precip_1h",
        "precip_6h",
        "precip_24h",
        "snow_at_draw",
    ]
    for col in numeric_weather_cols:
        if col in calendar_weather.columns:
            calendar_weather[col] = pd.to_numeric(calendar_weather[col], errors="coerce")
            calendar_weather.loc[calendar_weather[col] < 0, col] = np.nan

    calendar_weather["is_raining"] = calendar_weather["is_raining"].fillna(False).astype(int)
    calendar_weather["is_snowing"] = calendar_weather["is_snowing"].fillna(False).astype(int)
    calendar_weather["month_sin"] = np.sin(2 * np.pi * calendar_weather["month"] / 12)
    calendar_weather["month_cos"] = np.cos(2 * np.pi * calendar_weather["month"] / 12)
    calendar_weather["day_sin"] = np.sin(2 * np.pi * calendar_weather["day_of_month"] / 31)
    calendar_weather["day_cos"] = np.cos(2 * np.pi * calendar_weather["day_of_month"] / 31)

    calendar_cols = ["month_sin", "month_cos", "day_sin", "day_cos", "year"]
    weather_cols = [
        "temp_at_draw",
        "humidity_at_draw",
        "wind_at_draw",
        "pressure_at_draw",
        "precip_at_draw",
        "precip_day_cumulative_at_draw",
        "precip_1h",
        "precip_6h",
        "precip_24h",
        "is_raining",
        "is_snowing",
    ]

    feature_frame = pd.concat(
        [
            calendar_weather[["round"] + calendar_cols + weather_cols],
            pd.get_dummies(calendar_weather[["season"]], prefix="season", dtype=float),
            pd.get_dummies(calendar_weather[["temp_bin", "humidity_bin"]], dummy_na=True, dtype=float),
        ],
        axis=1,
    )
    return feature_frame


def build_context_feature_block(
    target_rounds: pd.Series,
    weather_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if weather_df is None:
        weather_df = load_weather_context()

    calendar_weather = prepare_calendar_weather_frame(weather_df)
    context_df = pd.DataFrame({"round": target_rounds}).merge(calendar_weather, on="round", how="left")

    for col in context_df.columns:
        if col == "round":
            continue
        if context_df[col].dtype.kind in "biufc":
            fill_value = context_df[col].median() if context_df[col].notna().any() else 0
            context_df[col] = context_df[col].fillna(fill_value)
        else:
            context_df[col] = context_df[col].fillna(0)

    calendar_cols = [col for col in context_df.columns if col in ["month_sin", "month_cos", "day_sin", "day_cos", "year"] or col.startswith("season_")]
    weather_cols = [col for col in context_df.columns if col in ["temp_at_draw", "humidity_at_draw", "wind_at_draw", "pressure_at_draw", "precip_at_draw", "precip_day_cumulative_at_draw", "precip_1h", "precip_6h", "precip_24h", "is_raining", "is_snowing"] or col.startswith("temp_bin_") or col.startswith("humidity_bin_")]

    return context_df, calendar_cols, weather_cols
