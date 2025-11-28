from __future__ import annotations

from typing import Any, Iterable, List

import pandas as pd


def summarize_runs(
    run_payloads: Iterable[dict[str, Any]],
    metric_key: str = "test_metrics",
) -> pd.DataFrame:
    """
    Build a summary table from run payloads.

    Each payload is expected to contain:
    - 'name': run identifier
    - metric_key: dict with at least total_return, sharpe, max_drawdown

    Returns a pandas DataFrame.
    """
    rows: List[dict[str, Any]] = []
    for payload in run_payloads:
        name = payload.get("name", "")
        metrics = payload.get(metric_key, {}) or {}
        row = {
            "name": name,
            "total_return": metrics.get("total_return", float("nan")),
            "annualized_return": metrics.get("annualized_return", float("nan")),
            "sharpe": metrics.get("sharpe", float("nan")),
            "max_drawdown": metrics.get("max_drawdown", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows)



