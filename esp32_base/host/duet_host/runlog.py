"""The base's record: one JSON line per check or job, appended to esp32_base/runs/<name>.jsonl.
Lines are only ever appended, so the file is the history of what was measured, where and when."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def append(name: str, record: dict) -> Path:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{name}.jsonl"
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path


def pct(values, p: float) -> float:
    """p-th percentile (0..100) by nearest rank; values need not be sorted."""
    v = sorted(values)
    if not v:
        return float("nan")
    k = max(0, min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1)))))
    return v[k]
