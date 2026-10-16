"""The contract every tool function in this repository follows.

A tool is a plain function that an agent calls by name with keyword arguments, and gets a dict back:

    {"status": "ok", ...what it measured...}
    {"status": "error", "message": "..."}

`@tool` is what makes that true. It never lets an exception escape — a model that passes a column that
does not exist should be told so and given the chance to correct itself, not crash the run — and it
converts whatever the function returned (numpy scalars, DataFrames, Paths) into JSON-safe values, since
the result is rendered back into a prompt.

    @tool(requires=("lifelines",))
    def fit_cox_model(data, duration, event): ...

`requires` names optional third-party packages. When one is not installed the tool reports that, and
which extra to install, instead of raising an ImportError from deep inside the function. The names are
also read by `ml_core.specs`, so the catalogue an agent sees says which tools need what.
"""

from __future__ import annotations

import functools
import importlib.util
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Import name -> what to `pip install` when it differs.
INSTALL_NAMES = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "sksurv": "scikit-survival",
    "sklearn_crfsuite": "sklearn-crfsuite",
    "sentence_transformers": "sentence-transformers",
    "umap": "umap-learn",
    "hdbscan": "hdbscan",
}


def jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert numpy / pandas / Path values into plain JSON types; NaN and inf become None."""
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient="records"), _depth + 1)
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict(), _depth + 1)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), _depth + 1)
    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, _depth + 1) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return str(value)


def ok(**fields: Any) -> dict[str, Any]:
    """A successful result."""
    return {"status": "ok", **jsonable(fields)}


def fail(message: str, **fields: Any) -> dict[str, Any]:
    """A failed result the agent can read and act on."""
    return {"status": "error", "message": message, **jsonable(fields)}


class MissingDependency(RuntimeError):
    """An optional package a tool needs is not installed."""


def missing_packages(packages: tuple[str, ...]) -> list[str]:
    return [p for p in packages if importlib.util.find_spec(p) is None]


def tool(fn: Callable[..., Any] | None = None, *, requires: tuple[str, ...] = ()) -> Any:
    """Mark a function as an agent-callable tool. See the module docstring."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            lacking = missing_packages(requires)
            if lacking:
                names = " ".join(INSTALL_NAMES.get(p, p) for p in lacking)
                return fail(
                    f"{func.__name__} needs {', '.join(lacking)}, which is not installed.",
                    install=f"pip install {names}",
                )
            try:
                result = func(*args, **kwargs)
            except MissingDependency as error:
                return fail(str(error))
            except Exception as error:  # a tool reports; it never raises into an agent run
                return fail(
                    f"{type(error).__name__}: {error}",
                    tool=func.__name__,
                    trace=traceback.format_exc(limit=3).splitlines()[-3:],
                )
            if isinstance(result, dict) and "status" in result:
                return jsonable(result)
            return ok(**(result if isinstance(result, dict) else {"result": result}))

        wrapper.__tool__ = True  # type: ignore[attr-defined]
        wrapper.__requires__ = requires  # type: ignore[attr-defined]
        return wrapper

    return decorate(fn) if fn is not None else decorate


def is_tool(obj: Any) -> bool:
    return callable(obj) and getattr(obj, "__tool__", False)
