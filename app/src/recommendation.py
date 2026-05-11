from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR
from src.features import FEATURE_SET_ORDER, build_model_feature_bundle, create_one_hot, load_clean_data
from src.features.pattern_features import DEFAULT_PATTERN_SOURCE_COLS, add_draw_pattern_columns
from src.models.model_suite import build_model_builders, get_probability_matrix
from src.models.predict import probability_matrix_to_number_lists

DEFAULT_WINDOW = 20
DEFAULT_RANDOM_SEED = 42
DEFAULT_STRICT_CANDIDATE = {
    "feature_set": "base_plus_pattern",
    "model": "soft_voting_ensemble",
    "training_regime": "last_500",
}
DEFAULT_CONTEXT_CANDIDATE = {
    "feature_set": "full_feature_set",
    "model": "soft_voting_ensemble",
    "training_regime": "last_500",
}


@dataclass
class NextDrawRecommendation:
    candidate_name: str
    feature_set: str
    model: str
    training_regime: str
    next_round: int
    next_draw_date: pd.Timestamp
    ticket_top6: list[int]
    top12_numbers: list[int]
    probability_series: pd.Series
    probability_frame: pd.DataFrame
    weighted_tickets: list[list[int]]
    diversified_tickets: list[list[int]]
    assumptions: dict[str, str]


def _parse_training_regime(training_regime: str) -> int | None:
    if training_regime == "full_history":
        return None
    if training_regime.startswith("last_"):
        return int(training_regime.split("_", 1)[1])
    raise ValueError(f"Unsupported training regime: {training_regime}")


def infer_next_round_number(clean_df: pd.DataFrame | None = None) -> int:
    clean_df = clean_df if clean_df is not None else load_clean_data()
    return int(clean_df["round"].max()) + 1


def infer_next_draw_date(context_rounds: pd.Series | None = None, context_dates: pd.Series | None = None) -> pd.Timestamp:
    if context_dates is not None and len(context_dates) > 0:
        context_dates = pd.to_datetime(context_dates, errors="coerce").dropna()
        if not context_dates.empty:
            return context_dates.max() + pd.Timedelta(days=7)

    today = pd.Timestamp.today().normalize()
    weekday = today.weekday()
    days_until_saturday = (5 - weekday) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    return today + pd.Timedelta(days=days_until_saturday)


def build_next_base_feature_row(clean_df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    one_hot = create_one_hot(clean_df)
    if len(one_hot) < window:
        raise ValueError(f"Need at least {window} rows to build base features; found {len(one_hot)}")

    recent_freq = one_hot.tail(window).sum(axis=0)

    gaps = {}
    next_index = len(one_hot)
    for num in one_hot.columns:
        hits = np.flatnonzero(one_hot[num].to_numpy() == 1)
        gaps[num] = 999 if len(hits) == 0 else int(next_index - hits[-1])

    row = {}
    for num in one_hot.columns:
        row[f"freq_{num}"] = float(recent_freq.loc[num])
    for num in one_hot.columns:
        row[f"gap_{num}"] = float(gaps[num])

    return pd.DataFrame([row])


def build_next_pattern_feature_row(
    clean_df: pd.DataFrame,
    rolling_windows: tuple[int, ...] = (5, 10, 20),
    source_cols: list[str] | None = None,
) -> pd.DataFrame:
    source_cols = source_cols or DEFAULT_PATTERN_SOURCE_COLS
    draw_patterns = add_draw_pattern_columns(clean_df)

    row = {}
    for col in source_cols:
        series = draw_patterns[col].astype(float)
        row[f"{col}_prev"] = float(series.iloc[-1])
        for window in rolling_windows:
            tail = series.tail(window)
            row[f"{col}_mean_{window}"] = float(tail.mean()) if not tail.empty else 0.0
            std_value = float(tail.std()) if len(tail) > 1 else 0.0
            row[f"{col}_std_{window}"] = 0.0 if np.isnan(std_value) else std_value

    return pd.DataFrame([row])


def _season_from_date(draw_date: pd.Timestamp) -> str:
    month = int(draw_date.month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_imputed_next_context_row(feature_bundle, next_round: int, next_draw_date: pd.Timestamp) -> pd.DataFrame:
    context_df = feature_bundle.context.copy()
    context_cols = [col for col in context_df.columns if col != "round"]
    row = pd.Series(index=context_cols, dtype=float)

    for col in context_cols:
        series = context_df[col]
        if pd.api.types.is_numeric_dtype(series):
            row[col] = float(series.mean()) if col.startswith(("season_", "temp_bin_", "humidity_bin_")) else float(series.median())
        else:
            row[col] = 0.0

    row["month_sin"] = float(np.sin(2 * np.pi * next_draw_date.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * next_draw_date.month / 12))
    row["day_sin"] = float(np.sin(2 * np.pi * next_draw_date.day / 31))
    row["day_cos"] = float(np.cos(2 * np.pi * next_draw_date.day / 31))
    row["year"] = float(next_draw_date.year)

    season = _season_from_date(next_draw_date)
    for season_col in [col for col in context_cols if col.startswith("season_")]:
        row[season_col] = 1.0 if season_col == f"season_{season}" else 0.0

    return pd.DataFrame([row]).fillna(0.0)


def build_next_feature_row(
    feature_bundle,
    feature_set_name: str,
    next_draw_date: pd.Timestamp | None = None,
    window: int = DEFAULT_WINDOW,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if feature_set_name not in FEATURE_SET_ORDER:
        raise ValueError(f"Unsupported feature set: {feature_set_name}")

    clean_df = feature_bundle.clean_df.copy()
    next_round = infer_next_round_number(clean_df)
    if next_draw_date is None:
        next_draw_date = infer_next_draw_date(
            context_rounds=feature_bundle.context.get("round"),
            context_dates=feature_bundle.context.get("draw_date") if "draw_date" in feature_bundle.context.columns else None,
        )

    base_row = build_next_base_feature_row(clean_df, window=window)
    assumptions = {
        "next_round": str(next_round),
        "next_draw_date": str(next_draw_date.date()),
        "feature_set": feature_set_name,
    }

    if feature_set_name == "base":
        return base_row, assumptions

    pattern_row = build_next_pattern_feature_row(clean_df)
    if feature_set_name == "base_plus_pattern":
        return pd.concat([base_row, pattern_row], axis=1), assumptions

    context_row = build_imputed_next_context_row(feature_bundle, next_round, next_draw_date)
    assumptions["unknown_future_weather"] = "imputed_from_historical_context"

    if feature_set_name == "base_plus_context":
        return pd.concat([base_row, context_row], axis=1), assumptions

    return pd.concat([base_row, pattern_row, context_row], axis=1), assumptions


def fit_candidate_model(
    feature_set: str,
    model_name: str,
    training_regime: str,
    window: int = DEFAULT_WINDOW,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    feature_bundle = build_model_feature_bundle(window=window, save_base=False)
    X = feature_bundle.feature_sets[feature_set].reset_index(drop=True)
    y = feature_bundle.y.reset_index(drop=True)

    train_window_size = _parse_training_regime(training_regime)
    if train_window_size is not None:
        X_train = X.tail(train_window_size).reset_index(drop=True)
        y_train = y.tail(train_window_size).reset_index(drop=True)
    else:
        X_train = X
        y_train = y

    builders = build_model_builders(random_seed)
    if model_name not in builders:
        raise ValueError(f"Unsupported model for recommendation: {model_name}")

    model = builders[model_name]()
    model.fit(X_train, y_train)
    return model, feature_bundle


def generate_weighted_tickets(probability_vector: np.ndarray, n_tickets: int = 5, seed: int = DEFAULT_RANDOM_SEED) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    weights = np.asarray(probability_vector, dtype=float)
    weights = np.clip(weights, a_min=1e-9, a_max=None)
    weights = weights / weights.sum()

    tickets = []
    for _ in range(n_tickets):
        picked = rng.choice(np.arange(1, 46), size=6, replace=False, p=weights)
        tickets.append(sorted(int(v) for v in picked))
    return tickets


def generate_diversified_tickets(probability_vector: np.ndarray, top_pool_size: int = 12, n_tickets: int = 5) -> list[list[int]]:
    ranked = np.argsort(probability_vector)[::-1] + 1
    pool = ranked[:top_pool_size]
    tickets = []
    for idx in range(n_tickets):
        rotated = np.roll(pool, -idx)
        tickets.append(sorted(int(v) for v in rotated[:6]))
    return tickets


def recommend_next_draw(
    candidate_name: str,
    feature_set: str,
    model_name: str,
    training_regime: str,
    next_draw_date: pd.Timestamp | None = None,
    window: int = DEFAULT_WINDOW,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_weighted_tickets: int = 5,
    n_diversified_tickets: int = 5,
) -> NextDrawRecommendation:
    model, feature_bundle = fit_candidate_model(
        feature_set=feature_set,
        model_name=model_name,
        training_regime=training_regime,
        window=window,
        random_seed=random_seed,
    )

    X_next, assumptions = build_next_feature_row(
        feature_bundle,
        feature_set_name=feature_set,
        next_draw_date=next_draw_date,
        window=window,
    )
    probability_matrix = get_probability_matrix(model, X_next)
    probability_vector = probability_matrix[0]
    ticket_top6 = probability_matrix_to_number_lists(probability_matrix)[0]

    probability_series = pd.Series(probability_vector, index=np.arange(1, 46), name="predicted_probability")
    probability_frame = (
        probability_series.sort_values(ascending=False)
        .rename_axis("number")
        .reset_index()
    )
    top12_numbers = probability_frame.head(12)["number"].astype(int).tolist()

    next_round = infer_next_round_number(feature_bundle.clean_df)
    if next_draw_date is None:
        next_draw_date = pd.Timestamp(assumptions["next_draw_date"])

    return NextDrawRecommendation(
        candidate_name=candidate_name,
        feature_set=feature_set,
        model=model_name,
        training_regime=training_regime,
        next_round=next_round,
        next_draw_date=pd.Timestamp(next_draw_date),
        ticket_top6=ticket_top6,
        top12_numbers=top12_numbers,
        probability_series=probability_series,
        probability_frame=probability_frame,
        weighted_tickets=generate_weighted_tickets(probability_vector, n_tickets=n_weighted_tickets, seed=random_seed),
        diversified_tickets=generate_diversified_tickets(probability_vector, n_tickets=n_diversified_tickets),
        assumptions=assumptions,
    )


def export_next_draw_outputs(
    recommendations: list[NextDrawRecommendation],
    figure_path: Path | None = None,
    summary_path: Path | None = None,
    ranking_path: Path | None = None,
):
    summary_rows = []
    ranking_frames = []
    for rec in recommendations:
        summary_rows.append(
            {
                "candidate_name": rec.candidate_name,
                "feature_set": rec.feature_set,
                "model": rec.model,
                "training_regime": rec.training_regime,
                "next_round": rec.next_round,
                "next_draw_date": rec.next_draw_date.date().isoformat(),
                "ticket_top6": ", ".join(map(str, rec.ticket_top6)),
                "top12_numbers": ", ".join(map(str, rec.top12_numbers)),
                "weighted_ticket_1": ", ".join(map(str, rec.weighted_tickets[0])) if rec.weighted_tickets else "",
                "diversified_ticket_1": ", ".join(map(str, rec.diversified_tickets[0])) if rec.diversified_tickets else "",
                "assumptions": "; ".join(f"{k}={v}" for k, v in rec.assumptions.items()),
            }
        )
        ranking_frame = rec.probability_frame.copy()
        ranking_frame.insert(0, "candidate_name", rec.candidate_name)
        ranking_frames.append(ranking_frame)

    summary_df = pd.DataFrame(summary_rows)
    ranking_df = pd.concat(ranking_frames, ignore_index=True)

    if summary_path is None:
        summary_path = TABLES_DIR / "table_42_next_draw_recommendations.csv"
    if ranking_path is None:
        ranking_path = TABLES_DIR / "table_43_next_draw_rankings.csv"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)

    return summary_df, ranking_df
