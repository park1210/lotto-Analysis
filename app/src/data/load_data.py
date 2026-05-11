from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_LOTTO_FILE
from src.data.sync_lotto_html import ensure_lotto_history_available


def resolve_lotto_excel_path(file_path=None) -> Path:
    return Path(file_path) if file_path is not None else RAW_LOTTO_FILE


def load_lotto_excel(file_path=RAW_LOTTO_FILE, sync: bool = True) -> pd.DataFrame:
    """
    Load lotto history from the existing local Excel file.
    """
    resolved_path = resolve_lotto_excel_path(file_path)
    if sync:
        ensure_lotto_history_available(output_path=resolved_path)
    return pd.read_excel(resolved_path)


def load_lotto_source(
    source: str = "excel",
    file_path=RAW_LOTTO_FILE,
    start_round: int = 1,
    end_round: int | None = None,
    sync: bool = True,
) -> pd.DataFrame:
    """
    Load lotto history from the local canonical Excel file.
    """
    _ = (start_round, end_round)

    if source == "excel":
        return load_lotto_excel(file_path=file_path, sync=sync)

    raise ValueError("source must be 'excel'")
