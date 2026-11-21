"""The job queue: work that takes longer than a moment (a whole song, a cloud call) waits here and runs
one at a time on a worker thread. The panel and the board never wait for it; they ask for the state.
Every finished job is appended to runs/jobs.jsonl."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable, Dict, List

from . import runlog


class JobQueue:
    def __init__(self, run: Callable[[dict], dict], keep: int = 100):
        """run(job) does the work and returns a dict that is merged into the job (e.g. clip_out, steps)."""
        self._run = run
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._jobs: Dict[int, dict] = {}
        self._lock = threading.Lock()
        self._n = 0
        self._keep = keep
        self._th = threading.Thread(target=self._worker, name="duet-jobs", daemon=True)
        self._th.start()

    def submit(self, **fields) -> dict:
        with self._lock:
            self._n += 1
            job = {"id": self._n, "state": "queued", "created": time.strftime("%H:%M:%S"), **fields}
            self._jobs[job["id"]] = job
            for old in sorted(self._jobs)[:-self._keep]:
                if self._jobs[old]["state"] in ("done", "failed"):
                    del self._jobs[old]
        self._q.put(job)
        return dict(job)

    def list(self) -> List[dict]:
        with self._lock:
            return [dict(j) for _, j in sorted(self._jobs.items(), reverse=True)]

    def get(self, job_id: int) -> dict:
        with self._lock:
            return dict(self._jobs[job_id])

    def wait(self, job_id: int, timeout: float = 120.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            j = self.get(job_id)
            if j["state"] in ("done", "failed"):
                return j
            time.sleep(0.02)
        raise TimeoutError(f"job {job_id} still {self.get(job_id)['state']} after {timeout:.0f} s")

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            self._set(job["id"], state="running")
            t0 = time.perf_counter()
            try:
                result = self._run(self.get(job["id"]))
                self._set(job["id"], state="done", seconds=round(time.perf_counter() - t0, 3), **result)
            except Exception as e:
                self._set(job["id"], state="failed", seconds=round(time.perf_counter() - t0, 3),
                          error=f"{e}" if type(e).__name__.startswith("Block") or type(e).__name__.endswith("Error")
                          else f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=4))
            done = self.get(job["id"])
            runlog.append("jobs", {k: v for k, v in done.items() if k != "trace"})

    def _set(self, job_id: int, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)
