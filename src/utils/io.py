"""I/O helpers for saving and loading experiment results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize data to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(
    metrics: dict,
    experiment_name: str,
    results_dir: str | Path = "results",
    extra: dict | None = None,
) -> Path:
    """
    Save experiment metrics to results/<experiment_name>_<timestamp>.json.

    Parameters
    ----------
    metrics         : dict returned by compute_metrics.
    experiment_name : short descriptive name for the run.
    results_dir     : directory to write results into.
    extra           : any additional metadata to include.

    Returns
    -------
    Path to the saved file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    out_path = results_dir / filename

    payload = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)

    save_json(payload, out_path)
    print(f"Results saved to {out_path}")
    return out_path
