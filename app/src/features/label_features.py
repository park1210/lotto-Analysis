from __future__ import annotations

import pandas as pd


NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
LABEL_COLS = [f"y_{i}" for i in range(1, 46)]


def build_label_frame(
    clean_df: pd.DataFrame,
    number_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    number_cols = number_cols or NUMBER_COLS
    label_cols = label_cols or LABEL_COLS

    label_df = pd.DataFrame(0, index=clean_df.index, columns=label_cols)
    for idx in range(len(clean_df)):
        nums = clean_df.loc[idx, number_cols].tolist()
        for num in nums:
            label_df.loc[idx, f"y_{int(num)}"] = 1
    return label_df
