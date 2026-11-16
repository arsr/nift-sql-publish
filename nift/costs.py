"""Wall-clock and GPU cost log for each stage of a run."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class CostLog:
    def __init__(self, path: Path, gpu_price_per_hour: float):
        self.path, self.price = Path(path), gpu_price_per_hour
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}

    @contextmanager
    def stage(self, name: str, n_questions: int | None = None):
        t0 = time.time()
        yield
        self.add(name, time.time() - t0, n_questions)

    def add(self, name: str, seconds: float, n_questions: int | None = None) -> None:
        r = {"seconds": round(seconds, 2), "gpu_minutes": round(seconds / 60, 2), "usd": round(seconds / 3600 * self.price, 4)}
        if n_questions:
            r["ms_per_question"] = round(1000 * seconds / n_questions, 1)
        self.rows[name] = r
        self.path.write_text(json.dumps(self.rows, indent=1))

    def show(self) -> None:
        print(f"{'stage':42s} {'minutes':>8s} {'USD':>8s} {'ms/question':>12s}   (GPU at ${self.price}/hour)")
        for k, r in self.rows.items():
            print(f"{k:42s} {r['gpu_minutes']:8.2f} {r['usd']:8.4f} {str(r.get('ms_per_question', '')):>12s}")
