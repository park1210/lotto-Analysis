import pandas as pd

from src.config import FEATURE_LOTTO_FILE, PROCESSED_LOTTO_FILE
from src.features.gap_features import create_gap_feature
from src.features.rolling_features import create_recent_frequency


number_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]


def load_clean_data():
    return pd.read_csv(PROCESSED_LOTTO_FILE)


def create_one_hot(df):
    one_hot = pd.DataFrame(0, index=df.index, columns=range(1, 46))

    for col in number_cols:
        for i, num in enumerate(df[col]):
            one_hot.loc[i, num] = 1

    return one_hot


def build_feature_dataset(window: int = 20, save: bool = True) -> pd.DataFrame:
    df = load_clean_data()
    one_hot = create_one_hot(df)

    recent_freq = create_recent_frequency(one_hot, window=window)
    gap = create_gap_feature(one_hot)

    valid_mask = recent_freq.notna().all(axis=1)

    recent_freq = recent_freq.loc[valid_mask].reset_index(drop=True)
    gap = gap.loc[valid_mask].reset_index(drop=True)

    gap = gap.fillna(999)

    feature_df = pd.concat([recent_freq.add_prefix("freq_"), gap.add_prefix("gap_")], axis=1)

    if save:
        FEATURE_LOTTO_FILE.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(FEATURE_LOTTO_FILE, index=False)

    return feature_df
