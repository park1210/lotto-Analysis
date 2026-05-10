from __future__ import annotations

import sys
from pathlib import Path


def find_app_root(start: str | Path | None = None) -> Path:
    """
    Resolve the project app root in a notebook-safe way.

    The function walks upward from the provided path (or the current working
    directory) until it finds a directory that contains both `src` and `data`.
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    search_roots = [current, *current.parents]

    for candidate in search_roots:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate

    raise RuntimeError(
        "Could not locate the app root. Run this notebook inside the project "
        "workspace, such as app/notebooks or app."
    )


def bootstrap_src_path(start: str | Path | None = None) -> Path:
    """
    Add the app root to sys.path so notebooks can import from `src.*`.
    """
    app_root = find_app_root(start=start)
    app_root_str = str(app_root)
    if app_root_str not in sys.path:
        sys.path.insert(0, app_root_str)
    return app_root


def describe_notebook_environment(start: str | Path | None = None) -> dict[str, str]:
    """
    Return lightweight environment info that can be displayed in notebooks.
    """
    app_root = bootstrap_src_path(start=start)
    return {
        "app_root": str(app_root),
        "cwd": str(Path.cwd().resolve()),
    }
